import json
import os
from os.path import dirname, join

import httpx
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)


def is_enabled():
    return bool(os.getenv("BRAVE_API_KEY"))


search = {
    "type": "function",
    "function": {
        "name": "get_brave_results",
        "description": "Search the web using Brave Search API and return structured results in JSON format.",
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


def get_brave_results(query):
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return json.dumps({"provider": "brave", "query": query, "error": "BRAVE_API_KEY is not set"}, ensure_ascii=False)

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": 10,
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return json.dumps({"provider": "brave", "query": query, "error": str(e)}, ensure_ascii=False)

    items = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "source": item.get("meta_url", {}).get("hostname", ""),
        }
        for item in data.get("web", {}).get("results", [])[:10]
    ]
    return json.dumps({"provider": "brave", "query": query, "result": {"items": items}}, ensure_ascii=False)
