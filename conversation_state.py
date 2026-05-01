import base64
from dataclasses import dataclass, field
from mimetypes import guess_type
from typing import List

from openai.types.beta.threads import (
    ImageFileContentBlock,
    ImageURLContentBlock,
    ImageURL,
    TextContentBlock,
    Text,
)
from openai.types.responses import ResponseCodeInterpreterToolCall


ContentBlock = ImageFileContentBlock | ImageURLContentBlock | TextContentBlock | ResponseCodeInterpreterToolCall


def _safe_json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    return str(value)


@dataclass
class ChatMessage:
    role: str
    content: List[ContentBlock]
    files: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def _serialize_content_block(cont):
        if isinstance(cont, TextContentBlock):
            return {
                "type": "text",
                "text": cont.text.value,
            }

        if isinstance(cont, ImageURLContentBlock):
            url = cont.image_url.url
            if isinstance(url, str) and url.startswith("data:"):
                return None
            return {
                "type": "image_url",
                "url": url,
                "detail": getattr(cont.image_url, "detail", "auto"),
            }

        return None

    @staticmethod
    def _deserialize_content_block(payload):
        ctype = payload.get("type")
        if ctype == "text":
            return TextContentBlock(type="text", text=Text(value=payload.get("text", ""), annotations=[]))
        if ctype == "image_url":
            return ImageURLContentBlock(
                type="image_url",
                image_url=ImageURL(url=payload.get("url", ""), detail=payload.get("detail", "auto")),
            )
        return None

    def to_snapshot_dict(self):
        serialized_content = []
        for cont in self.content:
            serial = self._serialize_content_block(cont)
            if serial:
                serialized_content.append(serial)

        metadata = {}
        if isinstance(self.metadata, dict) and isinstance(self.metadata.get("token_usage"), dict):
            metadata["token_usage"] = _safe_json_value(self.metadata.get("token_usage"))

        return {
            "role": self.role,
            "content": serialized_content,
            "files": [],
            "metadata": metadata,
        }

    @classmethod
    def from_snapshot_dict(cls, payload):
        if not isinstance(payload, dict):
            payload = {}

        content = []
        for cont in payload.get("content", []):
            block = cls._deserialize_content_block(cont)
            if block:
                content.append(block)

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            role=payload.get("role", "user"),
            content=content,
            files=[],
            metadata=metadata,
        )


class ChatThread:
    def __init__(self, client):
        self.client = client
        self.messages = []
        self.thread_id = None

    def add_message(self, model, role, content, files=None, metadata={}):
        if isinstance(content, str):
            content = [TextContentBlock(type="text", text=Text(value=content, annotations=[]))]

        self.messages.append(ChatMessage(role, content, files, metadata))

        if self.thread_id and (model["api_mode"] != "assistant" or role != "assistant"):
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role=role,
                content=self.content_to_content_param(content),
                attachments=files,
            )

    def get_last_message(self):
        return self.messages[-1]

    def get_last_message_id(self):
        return len(self.messages) - 1

    def get_messages_after(self, id):
        return self.messages[(id + 1):]

    def get_thread_id(self):
        if self.thread_id:
            return self.thread_id

        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": msg.role,
                    "content": self.content_to_content_param(msg.content),
                    "attachments": msg.files,
                }
                for msg in self.messages
                if msg.role == "user" or msg.role == "assistant"
            ]
        )
        self.thread_id = thread.id
        return self.thread_id

    @staticmethod
    def content_to_content_param(content: List[ContentBlock]) -> List[dict]:
        content_param = []
        for block in content:
            if block.type == "text":
                content_param.append({
                    "type": block.type,
                    "text": block.text.value,
                })
            elif block.type == "image_file":
                content_param.append({
                    "type": block.type,
                    "image_file": {"file_id": block.image_file.file_id, "detail": block.image_file.detail},
                })
            elif block.type == "image_url":
                content_param.append({
                    "type": block.type,
                    "image_url": {"url": block.image_url.url, "detail": block.image_url.detail},
                })
            elif block.type == "code_interpreter_call":
                content_param.append({
                    "type": block.type,
                    "id": block.id,
                    "container_id": block.container_id,
                    "code": block.code,
                })
            else:
                raise ValueError(f"未知のコンテンツブロックの type: {block.type}")
        return content_param

    def to_snapshot_dict(self):
        return {
            "thread_id": self.thread_id,
            "messages": [m.to_snapshot_dict() for m in self.messages],
        }

    def load_snapshot_dict(self, payload):
        if not isinstance(payload, dict):
            return False

        messages_raw = payload.get("messages", [])
        if not isinstance(messages_raw, list):
            return False

        self.thread_id = payload.get("thread_id")
        self.messages = [ChatMessage.from_snapshot_dict(m) for m in messages_raw]
        return True


class ConversationManager:
    def __init__(self, clients, assistants):
        self.client = clients["openai"]
        self.thread = ChatThread(self.client)
        self.assistants = assistants
        self.response_id = None
        self.response_last_message_id = -1
        self.code_interpreter_file_ids = []

    def add_message(self, model, role, content, files=None, metadata={}):
        self.thread.add_message(model, role, content, files, metadata)

    def to_snapshot_dict(self):
        thread_payload = self.thread.to_snapshot_dict()
        return {
            "thread_id": thread_payload.get("thread_id"),
            "response_id": self.response_id,
            "response_last_message_id": self.response_last_message_id,
            "code_interpreter_file_ids": self.code_interpreter_file_ids,
            "messages": thread_payload.get("messages", []),
        }

    @classmethod
    def from_snapshot_dict(cls, payload, clients, assistants):
        if not isinstance(payload, dict):
            return None

        conversation = cls(clients, assistants)
        ok = conversation.thread.load_snapshot_dict({
            "thread_id": payload.get("thread_id"),
            "messages": payload.get("messages", []),
        })
        if not ok:
            return None

        conversation.response_id = payload.get("response_id")
        conversation.response_last_message_id = payload.get("response_last_message_id", -1)
        conversation.code_interpreter_file_ids = payload.get("code_interpreter_file_ids", [])
        if not isinstance(conversation.code_interpreter_file_ids, list):
            conversation.code_interpreter_file_ids = []
        return conversation

    def get_completion_messages(self, model, text_only=False):
        messages = []
        for msg in self.thread.messages:
            content = [cont for cont in msg.content if not isinstance(cont, (ImageFileContentBlock, ResponseCodeInterpreterToolCall))]

            if not model.get("support_vision", False):
                content = [cont for cont in content if isinstance(cont, TextContentBlock)]

            if text_only:
                content = "\n".join([cont.text.value for cont in content if isinstance(cont, TextContentBlock)])
            else:
                content = self.thread.content_to_content_param(content)

            messages.append({
                "role": "assistant" if msg.role == "assistant" else "system" if msg.role == "system" else "system" if msg.role == "developer" else "user",
                "content": content,
            })
        return messages

    def set_response_id(self, response_id):
        self.response_id = response_id
        self.response_last_message_id = self.thread.get_last_message_id()

    def add_code_interpreter_file_ids(self, file_ids):
        self.code_interpreter_file_ids += file_ids
        self.code_interpreter_file_ids = list(dict.fromkeys(self.code_interpreter_file_ids))
        return self.code_interpreter_file_ids

    def get_response_history(self, model, offset=0):
        def is_file_for(what_for, file):
            for t in file["tools"]:
                if t["type"] == what_for:
                    return True
            return False

        messages = []
        for msg in self.thread.get_messages_after(self.response_last_message_id + offset):
            content = msg.content
            if msg.role == "assistant":
                content = [cont for cont in msg.content if not isinstance(cont, ImageFileContentBlock)]

            if not model.get("support_vision", False):
                content = [cont for cont in content if isinstance(cont, TextContentBlock)]

            content = self.thread.content_to_content_param(content)

            inout = "output" if msg.role == "assistant" else "input"
            content = [
                {
                    "text": cont["text"],
                    "type": inout + "_text",
                } if cont["type"] == "text" else
                {
                    "image_url": cont["image_url"]["url"],
                    "type": "input_image",
                } if cont["type"] == "image_url" else
                {
                    "file_id": cont["image_file"]["file_id"],
                    "type": "input_image",
                } if cont["type"] == "image_file" else
                cont
                for cont in content
            ]

            if msg.files:
                content += [
                    {
                        "file_id": file["file_id"],
                        "type": "input_file",
                    }
                    for file in msg.files if is_file_for("file_search", file)
                ]

            messages.append({
                "role": "assistant" if msg.role == "assistant" else "system" if msg.role == "system" else "developer" if msg.role == "developer" else "user",
                "content": content,
            })

            file_ids_for_code_interpreter = [
                file["file_id"]
                for file in msg.files if is_file_for("code_interpreter", file)
            ] if msg.files else []
            file_ids_for_code_interpreter = self.add_code_interpreter_file_ids(file_ids_for_code_interpreter)

        return messages, self.response_id, file_ids_for_code_interpreter

    def create_attachments(self, files, tool_for_files):
        attachments = []
        for file in files:
            file.seek(0)
            response = self.client.files.create(
                file=file,
                purpose="assistants",
            )
            attachments.append(
                {
                    "file_id": response.id,
                    "tools": [{"type": tool_for_files}],
                }
            )
        return attachments

    def create_ImageURLContentBlock(self, file, detail_level):
        mime_type = guess_type(file.name)[0]
        image_encoded = base64.b64encode(file.getvalue()).decode()
        image_url = f"data:{mime_type};base64,{image_encoded}"
        return ImageURLContentBlock(
            type="image_url",
            image_url=ImageURL(url=image_url, detail=detail_level),
        )
