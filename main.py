# import json
# import os
# import re
# from datetime import date, datetime
# from typing import Any, Dict, List, Optional

# from dateutil import parser as date_parser
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from openai import OpenAI

# app = FastAPI(title="Dynamic Schema Structured Extraction API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# class ExtractRequest(BaseModel):
#     text: str
#     schema: Dict[str, str]


# SUPPORTED_TYPES = {
#     "string",
#     "integer",
#     "float",
#     "boolean",
#     "date",
#     "array[string]",
#     "array[integer]",
# }


# def normalize_type(type_name: str) -> str:
#     return type_name.strip().lower()


# def parse_boolean(value: Any) -> Optional[bool]:
#     if value is None:
#         return None
#     if isinstance(value, bool):
#         return value
#     s = str(value).strip().lower()
#     if s in {"true", "yes", "1"}:
#         return True
#     if s in {"false", "no", "0"}:
#         return False
#     return None


# def parse_date(value: Any) -> Optional[str]:
#     if value is None:
#         return None
#     if isinstance(value, date):
#         return value.isoformat()
#     try:
#         dt = date_parser.parse(str(value), dayfirst=True, fuzzy=True)
#         return dt.date().isoformat()
#     except Exception:
#         return None


# def parse_int(value: Any) -> Optional[int]:
#     if value is None:
#         return None
#     try:
#         if isinstance(value, bool):
#             return None
#         if isinstance(value, int):
#             return value
#         if isinstance(value, float):
#             return int(value)
#         s = str(value).strip().replace(",", "")
#         match = re.search(r"-?\d+", s)
#         if match:
#             return int(match.group())
#         return None
#     except Exception:
#         return None


# def parse_float(value: Any) -> Optional[float]:
#     if value is None:
#         return None
#     try:
#         if isinstance(value, bool):
#             return None
#         if isinstance(value, (int, float)):
#             return float(value)
#         s = str(value).strip().replace(",", "")
#         match = re.search(r"-?\d+(\.\d+)?", s)
#         if match:
#             return float(match.group())
#         return None
#     except Exception:
#         return None


# def parse_array_string(value: Any) -> Optional[List[str]]:
#     if value is None:
#         return None
#     if isinstance(value, list):
#         return [str(v) for v in value]
#     return [str(value)]


# def parse_array_integer(value: Any) -> Optional[List[int]]:
#     if value is None:
#         return None
#     if isinstance(value, list):
#         result = []
#         for item in value:
#             parsed = parse_int(item)
#             if parsed is not None:
#                 result.append(parsed)
#         return result if result else None
#     parsed = parse_int(value)
#     return [parsed] if parsed is not None else None


# def coerce_value(value: Any, type_name: str) -> Any:
#     t = normalize_type(type_name)

#     if value is None:
#         return None

#     if t == "string":
#         return str(value)
#     if t == "integer":
#         return parse_int(value)
#     if t == "float":
#         return parse_float(value)
#     if t == "boolean":
#         return parse_boolean(value)
#     if t == "date":
#         return parse_date(value)
#     if t == "array[string]":
#         return parse_array_string(value)
#     if t == "array[integer]":
#         return parse_array_integer(value)

#     return None


# def validate_schema(schema: Dict[str, str]) -> None:
#     if not isinstance(schema, dict) or not schema:
#         raise HTTPException(status_code=400, detail="schema must be a non-empty object")

#     for field_name, type_name in schema.items():
#         if not isinstance(field_name, str) or not field_name.strip():
#             raise HTTPException(status_code=400, detail="schema keys must be non-empty strings")
#         if normalize_type(type_name) not in SUPPORTED_TYPES:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported type for field '{field_name}': {type_name}"
#             )


# def build_prompt(text: str, schema: Dict[str, str]) -> str:
#     schema_json = json.dumps(schema, indent=2)

#     return f"""
# You are a strict information extraction engine.

# Extract data from the given text according to the schema.

# RULES:
# 1. Return ONLY a valid JSON object.
# 2. Return EXACTLY the same keys as in the schema.
# 3. Do NOT add extra keys.
# 4. If a value cannot be confidently found, return null.
# 5. Use these output types:
#    - string -> JSON string
#    - integer -> JSON integer
#    - float -> JSON number
#    - boolean -> true or false
#    - date -> ISO format YYYY-MM-DD
#    - array[string] -> JSON array of strings
#    - array[integer] -> JSON array of integers
# 6. Do not explain anything.

# TEXT:
# {text}

# SCHEMA:
# {schema_json}
# """.strip()


# def call_llm_for_extraction(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
#     prompt = build_prompt(text, schema)

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": "You return only strict JSON."},
#                 {"role": "user", "content": prompt},
#             ],
#         )

#         content = response.choices[0].message.content.strip()

#         if content.startswith("```"):
#             content = re.sub(r"^```(?:json)?", "", content).strip()
#             content = re.sub(r"```$", "", content).strip()

#         parsed = json.loads(content)

#         if not isinstance(parsed, dict):
#             return {}

#         return parsed

#     except Exception:
#         return {}


# @app.get("/")
# def health():
#     return {"status": "ok"}


# @app.post("/dynamic-extract")
# def dynamic_extract(req: ExtractRequest):
#     validate_schema(req.schema)

#     raw = call_llm_for_extraction(req.text, req.schema)

#     result = {}
#     for field_name, type_name in req.schema.items():
#         raw_value = raw.get(field_name, None)
#         result[field_name] = coerce_value(raw_value, type_name)

#     return result

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}