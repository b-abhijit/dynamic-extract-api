import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI(title="Dynamic Extract API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]


SUPPORTED_TYPES = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "array[string]",
    "array[integer]",
}


def normalize_type(type_name: str) -> str:
    return type_name.strip().lower()


def validate_schema(schema: Dict[str, str]) -> None:
    if not isinstance(schema, dict) or not schema:
        raise HTTPException(status_code=400, detail="schema must be a non-empty object")

    for field_name, type_name in schema.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise HTTPException(status_code=400, detail="schema keys must be non-empty strings")

        normalized = normalize_type(type_name)
        if normalized not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type for field '{field_name}': {type_name}"
            )


def parse_boolean(value: Any) -> Optional[bool]:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()

    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False

    return None


def parse_integer(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    s = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+", s)
    if match:
        try:
            return int(match.group())
        except Exception:
            return None

    return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if match:
        try:
            return float(match.group())
        except Exception:
            return None

    return None


def parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, date):
        return value.isoformat()

    try:
        dt = date_parser.parse(str(value), dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None


def parse_array_string(value: Any) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, list):
        return [str(v) for v in value]

    return [str(value)]


def parse_array_integer(value: Any) -> Optional[List[int]]:
    if value is None:
        return None

    if isinstance(value, list):
        parsed_items = []
        for item in value:
            parsed = parse_integer(item)
            if parsed is not None:
                parsed_items.append(parsed)
        return parsed_items if parsed_items else None

    parsed = parse_integer(value)
    return [parsed] if parsed is not None else None


def coerce_value(value: Any, type_name: str) -> Any:
    normalized = normalize_type(type_name)

    if value is None:
        return None

    if normalized == "string":
        return str(value)

    if normalized == "integer":
        return parse_integer(value)

    if normalized == "float":
        return parse_float(value)

    if normalized == "boolean":
        return parse_boolean(value)

    if normalized == "date":
        return parse_date(value)

    if normalized == "array[string]":
        return parse_array_string(value)

    if normalized == "array[integer]":
        return parse_array_integer(value)

    return None


def build_prompt(text: str, schema: Dict[str, str]) -> str:
    schema_json = json.dumps(schema, indent=2)

    return f"""
You are a strict information extraction engine.

Extract data from the text according to the schema.

Rules:
1. Return ONLY a valid JSON object.
2. Return EXACTLY the same keys as the schema.
3. Do NOT add extra keys.
4. If a value cannot be found confidently, return null.
5. Required output types:
   - string -> JSON string
   - integer -> JSON integer
   - float -> JSON number
   - boolean -> true or false
   - date -> ISO format YYYY-MM-DD
   - array[string] -> JSON array of strings
   - array[integer] -> JSON array of integers
6. No explanation. No markdown. No code fences.

Text:
{text}

Schema:
{schema_json}
""".strip()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def call_llm(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    client = get_openai_client()
    prompt = build_prompt(text, schema)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Return only strict JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        parsed = json.loads(content)

        if isinstance(parsed, dict):
            return parsed

        return {}

    except Exception:
        return {}


@app.get("/")
def root():
    return {"ok": True, "message": "Dynamic extract API is running"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_key_present": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.post("/dynamic-extract")
def dynamic_extract(req: ExtractRequest):
    validate_schema(req.schema)

    raw_output = call_llm(req.text, req.schema)

    result = {}
    for field_name, type_name in req.schema.items():
        raw_value = raw_output.get(field_name, None)
        result[field_name] = coerce_value(raw_value, type_name)

    return result