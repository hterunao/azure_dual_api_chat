import json
import os
from os.path import dirname, join

import httpx
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)


def is_enabled():
    return bool(os.getenv("TAVILY_API_KEY"))


search = {
    "type": "function",
    "function": {
        "name": "get_tavily_results",
        "description": "Search the web using Tavily API and return structured results in JSON format.",
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


def get_tavily_results(query):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"provider": "tavily", "query": query, "error": "TAVILY_API_KEY is not set"}, ensure_ascii=False)

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 10,
        "include_answer": True,
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return json.dumps({"provider": "tavily", "query": query, "error": str(e)}, ensure_ascii=False)

    items = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "score": item.get("score"),
        }
        for item in data.get("results", [])[:10]
    ]
    return json.dumps(
        {
            "provider": "tavily",
            "query": query,
            "answer": data.get("answer"),
            "result": {"items": items},
        },
        ensure_ascii=False,
    )
