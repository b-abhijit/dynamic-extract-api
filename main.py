import re
from datetime import date
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Dynamic Extract API - No OpenAI")


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


INVALID_NAME_VALUES = {
    "employee",
    "customer",
    "name",
    "client",
    "staff",
    "buyer",
    "seller",
    "person",
    "employee name",
    "customer name",
    "patient",
    "doctor",
}


def normalize_type(type_name: str) -> str:
    return type_name.strip().lower()


def validate_schema(schema_map: Dict[str, str]) -> None:
    if not isinstance(schema_map, dict) or not schema_map:
        raise HTTPException(status_code=400, detail="schema must be a non-empty object")

    for field_name, type_name in schema_map.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise HTTPException(status_code=400, detail="schema keys must be non-empty strings")

        if normalize_type(type_name) not in SUPPORTED_TYPES:
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
    t = normalize_type(type_name)

    if value is None:
        return None
    if t == "string":
        return str(value)
    if t == "integer":
        return parse_integer(value)
    if t == "float":
        return parse_float(value)
    if t == "boolean":
        return parse_boolean(value)
    if t == "date":
        return parse_date(value)
    if t == "array[string]":
        return parse_array_string(value)
    if t == "array[integer]":
        return parse_array_integer(value)

    return None


def clean_extracted_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().rstrip(".,:;-")


def is_valid_person_name(name: str) -> bool:
    cleaned = clean_extracted_name(name)
    if not cleaned:
        return False
    if cleaned.lower() in INVALID_NAME_VALUES:
        return False
    if len(cleaned.split()) < 2:
        return False
    return True


def extract_person_name_by_label(text: str, labels: List[str]) -> Optional[str]:
    for label in labels:
        pattern = rf'\b{re.escape(label)}\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = clean_extracted_name(match.group(1))
            if is_valid_person_name(candidate):
                return candidate
    return None


def extract_person_name_generic(text: str) -> Optional[str]:
    patterns = [
        r'\bname\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bpatient\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bdoctor\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bemployee\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bcustomer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bclient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bbuyer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = clean_extracted_name(match.group(1))
            if is_valid_person_name(candidate):
                return candidate

    matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    blacklist = {
        "Employee Name",
        "Customer Name",
        "Order Date",
        "Total Amount",
        "Purchase Date",
        "Patient Name",
        "Doctor Name",
    }

    for candidate in matches:
        cleaned = clean_extracted_name(candidate)
        if cleaned not in blacklist and is_valid_person_name(cleaned):
            return cleaned

    return None


def extract_money(text: str) -> Optional[float]:
    patterns = [
        r'Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)',
        r'INR\s*([0-9,]+(?:\.[0-9]+)?)',
        r'\$\s*([0-9,]+(?:\.[0-9]+)?)',
        r'Total:\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)',
        r'amount[:\s]+\$?\s*([0-9,]+(?:\.[0-9]+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))
    return None


def extract_salary(text: str) -> Optional[float]:
    patterns = [
        r'\bmonthly salary\s*[:\-]\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bmonthly salary\s*[:\-]\s*INR\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bmonthly salary\s*[:\-]\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bsalary\s*[:\-]\s*Rs\.?\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bsalary\s*[:\-]\s*INR\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bsalary\s*[:\-]\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bpay\s*[:\-]\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\b',
        r'\bcompensation\s*[:\-]\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))

    return None


def extract_power_kw(text: str) -> Optional[float]:
    patterns = [
        r'\bpower[_\s-]*kw\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\b',
        r'\bpower\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\b',
        r'\bload\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\b',
        r'\bcapacity\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*k\s*w\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))

    return None


def extract_energy_kwh(text: str) -> Optional[float]:
    patterns = [
        r'\benergy[_\s-]*kwh\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\b',
        r'\benergy\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\s*h\b',
        r'\bconsumption\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\s*h\b',
        r'\busage\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*k\s*w\s*h\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*k\s*w\s*h\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))

    return None


def extract_hours(text: str) -> Optional[float]:
    patterns = [
        r'\bhours\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\b',
        r'\bduration\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*hours?\b',
        r'\bworked\s*([0-9]+(?:\.[0-9]+)?)\s*hours?\b',
        r'\bfor\s*([0-9]+(?:\.[0-9]+)?)\s*hours?\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*hours?\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*hrs\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))

    return None


def extract_metric(text: str) -> Optional[str]:
    patterns = [
        r'\bmetric\s*[:\-]\s*([A-Za-z][A-Za-z0-9 _/\-]+)',
        r'\bmetric_name\s*[:\-]\s*([A-Za-z][A-Za-z0-9 _/\-]+)',
        r'\balert\s+for\s+([A-Za-z][A-Za-z0-9 _/\-]+)',
        r'\b([A-Z]{2,}\s+usage)\b',
        r'\b([A-Za-z]+\s+usage)\b',
        r'\b([A-Za-z]+\s+utilization)\b',
        r'\b([A-Za-z]+\s+load)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip().rstrip(".,:;-")
            if value:
                return value

    return None


def extract_threshold(text: str) -> Optional[float]:
    patterns = [
        r'\bthreshold\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?\b',
        r'\bthreshold\s+(?:is|was|to|at)\s+([0-9]+(?:\.[0-9]+)?)\s*%?\b',
        r'\b(?:alert\s+)?threshold\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*%?\b',
        r'\btarget\s*[:=\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*%?\s*(?:threshold|target)\b',
        r'\b([0-9]+(?:\.[0-9]+)?)\s*%?\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_float(match.group(1))

    return None


def extract_time_hhmm(text: str) -> Optional[str]:
    patterns = [
        r'\balert[_\s-]*time\s*[:=\-]?\s*([0-2]?\d:[0-5]\d)\b',
        r'\btime\s*[:=\-]?\s*([0-2]?\d:[0-5]\d)\b',
        r'\b([0-2]?\d:[0-5]\d)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_host(text: str) -> Optional[str]:
    patterns = [
        r'\bhost\s*[:=\-]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,62})\b',
        r'\bhostname\s*[:=\-]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,62})\b',
        r'\bserver\s*[:=\-]?\s*([A-Za-z0-9][A-Za-z0-9-]{1,62})\b',
        r'\b([A-Za-z0-9][A-Za-z0-9-]{1,62})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".,:;-")
    return None


def extract_patient(text: str) -> Optional[str]:
    patterns = [
        r'\bpatient\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bpatient\s+name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
        r'\bpatient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = clean_extracted_name(match.group(1))
            if is_valid_person_name(candidate):
                return candidate
    return None


def extract_age(text: str) -> Optional[int]:
    patterns = [
        r'\bage\s*[:=\-]?\s*(\d{1,3})\b',
        r'\bpatient\s*age\s*[:=\-]?\s*(\d{1,3})\b',
        r'\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/o)\b',
        r'\bage\s+is\s+(\d{1,3})\b',
        r'\b(\d{1,3})\b(?=\s*(?:year|years|yrs|y/o)\b)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = parse_integer(match.group(1))
            if value is not None and 0 <= value <= 130:
                return value
    return None


def extract_date_value(text: str) -> Optional[str]:
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b',
        r'\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_date(match.group(0))
    return None


def extract_order_id(text: str) -> Optional[str]:
    patterns = [
        r'\b(?:Order\s*#?\s*)([A-Za-z0-9-]+)\b',
        r'\b(ORD-[A-Za-z0-9-]+)\b',
        r'\b([A-Z]{2,}-\d{3,})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_store(text: str) -> Optional[str]:
    patterns = [
        r'from\s+([A-Z][A-Za-z0-9& ]+)',
        r'store[:\s]+([A-Z][A-Za-z0-9& ]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def extract_city(text: str) -> Optional[str]:
    patterns = [
        r'Shipped to:\s*([A-Z][A-Za-z ]+)',
        r'city[:\s]+([A-Z][A-Za-z ]+)',
        r'in\s+([A-Z][A-Za-z ]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def extract_department(text: str) -> Optional[str]:
    patterns = [
        r'\bdepartment\s*[:\-]\s*([A-Z][A-Za-z&/ -]+)\b',
        r'\bdept\.?\s*[:\-]\s*([A-Z][A-Za-z&/ -]+)\b',
        r'\bdepartment\s+([A-Z][A-Za-z&/ -]+)\b',
        r'\bdept\.?\s+([A-Z][A-Za-z&/ -]+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip().rstrip(".,:;-")
            if value:
                return value

    known_departments = [
        "Engineering",
        "HR",
        "Human Resources",
        "Finance",
        "Marketing",
        "Sales",
        "Operations",
        "Support",
        "Product",
        "Legal",
        "IT",
        "Administration",
    ]

    for dept in known_departments:
        if re.search(rf'\b{re.escape(dept)}\b', text, re.IGNORECASE):
            return dept

    return None


def extract_grade(text: str) -> Optional[str]:
    patterns = [
        r'\bgrade\s*[:\-]\s*([A-Z]\d+)\b',
        r'\blevel\s*[:\-]\s*([A-Z]\d+)\b',
        r'\bband\s*[:\-]\s*([A-Z]\d+)\b',
        r'\bgrade\s+([A-Z]\d+)\b',
        r'\blevel\s+([A-Z]\d+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    fallback = re.search(r'\b([A-Z]\d+)\b', text)
    if fallback:
        return fallback.group(1).upper()

    return None


def extract_device(text: str) -> Optional[str]:
    patterns = [
        r'\bdevice\s*[:\-]\s*([A-Za-z][A-Za-z0-9 /_-]*\d+)\b',
        r'\basset\s*[:\-]\s*([A-Za-z][A-Za-z0-9 /_-]*\d+)\b',
        r'\bequipment\s*[:\-]\s*([A-Za-z][A-Za-z0-9 /_-]*\d+)\b',
        r'\bmachine\s*[:\-]\s*([A-Za-z][A-Za-z0-9 /_-]*\d+)\b',
        r'\bdevice\s+([A-Za-z][A-Za-z0-9 /_-]*\d+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".,:;-")

    fallback_patterns = [
        r'\b([A-Z]{2,}(?:\s+[A-Za-z0-9]+)*\s+\d+)\b',
        r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+\d+)\b',
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip(".,:;-")

    return None


def extract_quantity(text: str) -> Optional[int]:
    patterns = [
        r'\bquantity\s*[:\-]\s*(\d+)\b',
        r'\bcount\s*[:\-]\s*(\d+)\b',
        r'\b(\d+)\s+\w+',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return parse_integer(match.group(1))
    return None


def extract_item(text: str) -> Optional[str]:
    patterns = [
        r'bought\s+\d+\s+([A-Za-z ]+?)\s+for',
        r'Items:\s*(.+?)\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_boolean_value(text: str) -> Optional[bool]:
    if re.search(r'\b(true|yes)\b', text, re.IGNORECASE):
        return True
    if re.search(r'\b(false|no)\b', text, re.IGNORECASE):
        return False
    return None


def extract_array_strings(text: str) -> Optional[List[str]]:
    if "Items:" in text:
        after = text.split("Items:", 1)[1]
        after = after.split(".", 1)[0]
        parts = [p.strip() for p in after.split(" and ")]
        cleaned = []
        for part in parts:
            cleaned_part = re.sub(r'^\d+\s+', '', part).strip()
            if cleaned_part:
                cleaned.append(cleaned_part)
        return cleaned if cleaned else None
    return None


def extract_array_integers(text: str) -> Optional[List[int]]:
    nums = re.findall(r'\b\d+\b', text)
    values = [int(n) for n in nums]
    return values if values else None


def generic_extract(field_name: str, field_type: str, text: str) -> Any:
    name = field_name.lower()
    ftype = normalize_type(field_type)

    if "employee_name" in name:
        return extract_person_name_by_label(
            text,
            ["employee name", "employee", "staff name", "staff"]
        ) or extract_person_name_generic(text)

    if "customer_name" in name:
        return extract_person_name_by_label(
            text,
            ["customer name", "customer", "client name", "client", "buyer"]
        ) or extract_person_name_generic(text)

    if "patient" in name:
        return extract_patient(text) or extract_person_name_by_label(
            text,
            ["patient", "patient name"]
        ) or extract_person_name_generic(text)

    if "department" in name or name == "dept":
        return extract_department(text)

    if "monthly_salary" in name or "salary" in name or "pay" in name or "compensation" in name:
        return extract_salary(text)

    if "energy_kwh" in name or ("energy" in name and ftype == "float"):
        return extract_energy_kwh(text)

    if "power_kw" in name or ("power" in name and ftype == "float"):
        return extract_power_kw(text)

    if name == "hours" or "hours" in name or "duration" in name:
        return extract_hours(text)

    if name == "alert_time" or "time" in name:
        return extract_time_hhmm(text)

    if name == "host" or "hostname" in name or "server" in name:
        return extract_host(text)

    if name == "metric" or "metric" in name:
        return extract_metric(text)

    if "threshold" in name:
        return extract_threshold(text)

    if "age" in name:
        return extract_age(text)

    if "grade" in name or "level" in name or "band" in name:
        return extract_grade(text)

    if "device" in name or "asset" in name or "equipment" in name or "machine" in name:
        return extract_device(text)

    if name.endswith("_name") or name == "name":
        return extract_person_name_by_label(
            text,
            ["name", "full name"]
        ) or extract_person_name_generic(text)

    if "order" in name and "id" in name:
        return extract_order_id(text)

    if "date" in name:
        return extract_date_value(text)

    if "amount" in name or "price" in name or "total" in name:
        return extract_money(text)

    if "quantity" in name or "count" in name:
        return extract_quantity(text)

    if "store" in name or "shop" in name:
        return extract_store(text)

    if "city" in name or "location" in name:
        return extract_city(text)

    if "item" in name or "product" in name:
        return extract_item(text)

    if ftype == "boolean":
        return extract_boolean_value(text)

    if ftype == "array[string]":
        return extract_array_strings(text)

    if ftype == "array[integer]":
        return extract_array_integers(text)

    if ftype == "date":
        return extract_date_value(text)

    if ftype == "float":
        return extract_money(text)

    if ftype == "integer":
        return extract_quantity(text)

    return None


@app.get("/")
def root():
    return {"ok": True, "message": "Dynamic extract API is running without OpenAI"}


@app.get("/health")
def health():
    return {"ok": True, "mode": "rule-based"}


@app.post("/dynamic-extract")
def dynamic_extract(req: ExtractRequest):
    schema_map = req.schema_

    validate_schema(schema_map)

    result = {}
    for field_name, field_type in schema_map.items():
        guessed_value = generic_extract(field_name, field_type, req.text)
        result[field_name] = coerce_value(guessed_value, field_type)

    return result