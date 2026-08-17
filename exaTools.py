import json
import os
from os.path import dirname, join

import httpx
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)


def is_enabled():
    return bool(os.getenv("EXA_API_KEY"))


search = {
    "type": "function",
    "function": {
        "name": "get_exa_results",
        "description": "Search the web using Exa API and return structured results in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                }
            },
            "required": ["query"],
        },
    },
}


def get_exa_results(query):
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return json.dumps({"provider": "exa", "query": query, "error": "EXA_API_KEY is not set"}, ensure_ascii=False)

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "numResults": 10,
        "contents": {"text": {"maxCharacters": 800}},
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("https://api.exa.ai/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return json.dumps({"provider": "exa", "query": query, "error": str(e)}, ensure_ascii=False)

    items = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("text", ""),
            "published_date": item.get("publishedDate"),
            "score": item.get("score"),
        }
        for item in data.get("results", [])[:10]
    ]
    return json.dumps({"provider": "exa", "query": query, "result": {"items": items}}, ensure_ascii=False)
