import json
import os
from os.path import dirname, join

import httpx
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)


def is_enabled():
    return bool(os.getenv("SERPER_API_KEY"))


run = {
    "type": "function",
    "function": {
        "name": "get_google_serper",
        "description": "Perform a Google search to get latest information and get concise search results",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                },
            },
            "required": ["query"],
        },
    },
}

results = {
    "type": "function",
    "function": {
        "name": "get_google_results",
        "description": "Perform a Google search for latest or detailed information and get detailed results and metadata with JSON format",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                },
            },
            "required": ["query"],
        },
    },
}

scholar = {
    "type": "function",
    "function": {
        "name": "get_google_scholar",
        "description": "Get Google Scholar result for given search words in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                },
            },
            "required": ["query"],
        },
    },
}

news = {
    "type": "function",
    "function": {
        "name": "get_google_news",
        "description": "Get latest news for given search words in JSON format.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                },
            },
            "required": ["query"],
        },
    },
}

places = {
    "type": "function",
    "function": {
        "name": "get_google_places",
        "description": "Get latest places information (e.g. restaurants or shops or famous places) in JSON format for given search words",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words to search for",
                },
                "country": {
                    "type": "string",
                    "description": "Country where the place to search is located (ISO 3166-1 alpha-2, e.g. us, uk, jp)",
                },
                "language": {
                    "type": "string",
                    "description": "Language to express results (ISO639, e.g. en, ja)",
                },
            },
            "required": ["query"],
        },
    },
}


def _error_result(query, message):
    return json.dumps({"provider": "serper", "query": query, "error": message}, ensure_ascii=False)


def _post_serper(path, payload):
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return None, "SERPER_API_KEY is not set"

    url = f"https://google.serper.dev/{path}"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json(), None
    except Exception as e:
        return None, str(e)


def _compact_organic(data):
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": item.get("source", ""),
        }
        for item in data.get("organic", [])[:10]
    ]


def get_google_serper(query):
    data, error = _post_serper("search", {"q": query})
    if error:
        return _error_result(query, error)
    concise = "\n".join([f"- {x['title']}: {x['url']}" for x in _compact_organic(data)[:5]])
    return json.dumps(
        {"provider": "serper", "query": query, "result": concise, "items": _compact_organic(data)},
        ensure_ascii=False,
    )


def get_google_results(query):
    data, error = _post_serper("search", {"q": query})
    if error:
        return _error_result(query, error)
    return json.dumps({"provider": "serper", "query": query, "result": data}, ensure_ascii=False)


def get_google_scholar(query):
    data, error = _post_serper("scholar", {"q": query})
    if error:
        return _error_result(query, error)
    return json.dumps({"provider": "serper", "query": query, "result": data}, ensure_ascii=False)


def get_google_news(query):
    data, error = _post_serper("news", {"q": query})
    if error:
        return _error_result(query, error)
    return json.dumps({"provider": "serper", "query": query, "result": data}, ensure_ascii=False)


def get_google_places(query, country, language):
    payload = {"q": query, "gl": country, "hl": language}
    data, error = _post_serper("places", payload)
    if error:
        return _error_result(query, error)
    return json.dumps({"provider": "serper", "query": query, "result": data}, ensure_ascii=False)
