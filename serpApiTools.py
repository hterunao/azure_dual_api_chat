import json
import os
from os.path import dirname, join

import httpx
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)


def is_enabled():
    return bool(os.getenv("SERPAPI_API_KEY"))


search = {
    "type": "function",
    "function": {
        "name": "get_serpapi_results",
        "description": "Search Google via SerpApi and return structured results in JSON format.",
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


def get_serpapi_results(query):
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return json.dumps({"provider": "serpapi", "query": query, "error": "SERPAPI_API_KEY is not set"}, ensure_ascii=False)

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 10,
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get("https://serpapi.com/search.json", params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return json.dumps({"provider": "serpapi", "query": query, "error": str(e)}, ensure_ascii=False)

    items = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": item.get("source", ""),
        }
        for item in data.get("organic_results", [])[:10]
    ]
    return json.dumps({"provider": "serpapi", "query": query, "result": {"items": items}}, ensure_ascii=False)
