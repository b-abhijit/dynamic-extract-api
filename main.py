import json
import os
import re
import traceback
from datetime import date
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


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
    schema_: Dict[str, str] = Field(..., alias="schema")

    class Config:
        allow_population_by_field_name = True


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


def validate_schema(schema_map: Dict[str, str]) -> None:
    print("validate_schema:start")

    if not isinstance(schema_map, dict) or not schema_map:
        raise HTTPException(status_code=400, detail="schema must be a non-empty object")

    for field_name, type_name in schema_map.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise HTTPException(status_code=400, detail="schema keys must be non-empty strings")

        normalized = normalize_type(type_name)
        if normalized not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type for field '{field_name}': {type_name}"
            )

    print("validate_schema:done")


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
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    s = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+", s)
    return int(match.group()) if match else None


def parse_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else None


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
        result = []
        for item in value:
            parsed = parse_integer(item)
            if parsed is not None:
                result.append(parsed)
        return result if result else None

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


def build_prompt(text: str, schema_map: Dict[str, str]) -> str:
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
{json.dumps(schema_map, indent=2)}
""".strip()


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    print("env_key_present:", bool(api_key))

    if not api_key:
        raise Exception("OPENAI_API_KEY is not set on the server")

    client = OpenAI(api_key=api_key)
    print("openai_client_created")
    return client


def call_llm(text: str, schema_map: Dict[str, str]) -> Dict[str, Any]:
    print("call_llm:start")
    client = get_openai_client()
    prompt = build_prompt(text, schema_map)
    print("prompt_built")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only strict JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    print("openai_response_received")

    content = response.choices[0].message.content
    print("raw_content_type:", type(content).__name__)
    print("raw_content_preview:", repr(str(content)[:500]))

    if content is None:
        raise Exception("Model returned empty content")

    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    parsed = json.loads(content)
    print("json_parsed_successfully")

    if not isinstance(parsed, dict):
        raise Exception("LLM did not return a JSON object")

    return parsed


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
    try:
        print("dynamic_extract:start")
        schema_map = req.schema_
        print("schema_keys:", list(schema_map.keys()))

        validate_schema(schema_map)
        raw_output = call_llm(req.text, schema_map)

        result = {}
        for field_name, type_name in schema_map.items():
            raw_value = raw_output.get(field_name, None)
            result[field_name] = coerce_value(raw_value, type_name)

        print("dynamic_extract:success")
        return result

    except HTTPException as e:
        print("http_exception:", str(e.detail))
        raise e

    except Exception as e:
        trace = traceback.format_exc()
        print("unhandled_exception:", str(e))
        print(trace)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "trace": trace
            }
        )