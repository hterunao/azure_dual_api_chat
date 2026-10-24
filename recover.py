import json
import os

from dotenv import load_dotenv
from openai import OpenAI, APIError

load_dotenv(".env.local")

api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["ENDPOINT_URL"].rstrip("/")

# ENDPOINT_URL がリソースのルートURLでも、/openai/v1/ まで含んでいても対応
base_url = (
    endpoint + "/"
    if endpoint.endswith("/openai/v1")
    else endpoint + "/openai/v1/"
)

client = OpenAI(base_url=base_url, api_key=api_key)

# ここを書き換える：msg_... ではなく resp_... が必要
START_RESPONSE_ID = "resp_07a8003f48f049b8006adb9c1a5dc08193b70b9e3bbc272a0f"
OUTPUT_FILE = "recovered_responses.json"

records = []
seen = set()
response_id = START_RESPONSE_ID
max_responses = 1000  # 意図しない無限ループを防ぐ上限

while response_id and len(records) < max_responses:
    if response_id in seen:
        print(f"同じIDが再登場したため停止: {response_id}")
        break
    seen.add(response_id)

    try:
        response = client.responses.retrieve(response_id)
    except APIError as e:
        print(f"Responseを取得できませんでした: {response_id}")
        print(f"{type(e).__name__}: {e}")
        break

    record = {
        "response": response.model_dump(mode="json"),
        "input_items": [],
    }

    # そのResponseに送信された入力も取得する（複数ページなら順に取得）
    try:
        for item in client.responses.input_items.list(response.id):
            record["input_items"].append(item.model_dump(mode="json"))
    except APIError as e:
        record["input_items_error"] = f"{type(e).__name__}: {e}"
        print(f"入力アイテムの取得に失敗: {response.id}（Response本体は保存します）")

    records.append(record)
    print(
        f"取得: {response.id} "
        f"status={response.status} "
        f"previous={response.previous_response_id}"
    )

    response_id = response.previous_response_id

# 古い順に並べて保存。各レコードにResponse本体と入力アイテムを保持
records.reverse()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\n{len(records)}件を {OUTPUT_FILE} に保存しました。")
print("※ JSONには入力・出力など機密情報が含まれる可能性があります。")

def message_text(message):
    """messageのcontentからテキスト部分だけ取り出す"""
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") in ("input_text", "output_text")
        and isinstance(part.get("text"), str)
    )


print("\n========== 対話履歴（古い順） ==========")

seen_user_ids = set()
user_count = 0
assistant_count = 0

for number, record in enumerate(records, start=1):
    response = record["response"]
    print(f"\n--- Response {number}: {response['id']} ---")

    # input_itemsには過去の入力が再登場することがあるため、IDで重複を除く
    new_user_messages = []
    for item in record["input_items"]:
        if item.get("type") != "message" or item.get("role") != "user":
            continue

        message_id = item.get("id")
        if message_id:
            if message_id in seen_user_ids:
                continue
            seen_user_ids.add(message_id)

        text = message_text(item)
        if text:
            new_user_messages.append(text)

    for text in new_user_messages:
        user_count += 1
        print(f"ユーザー: {text}")

    # 回答は、そのResponseが生成したoutputだけを見る
    found_assistant = False
    for item in response.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue

        text = message_text(item)
        if text:
            assistant_count += 1
            found_assistant = True
            print(f"アシスタント: {text}")

    if not new_user_messages and not found_assistant:
        print("（表示できるテキストメッセージなし：ツール呼び出し等の可能性）")

print(
    f"\nユーザー入力 {user_count}件 / "
    f"アシスタントのテキストメッセージ {assistant_count}件"
)

