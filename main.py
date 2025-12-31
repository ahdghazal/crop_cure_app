import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# Optional OpenAI (used only if /explain endpoint is called)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()

DIAGNOSIS_CREATE_URL = os.getenv(
    "DIAGNOSIS_CREATE_URL",
    "https://crop.kindwise.com/api/v1/identification",
)
DIAGNOSIS_API_KEY = os.getenv("DIAGNOSIS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DIAGNOSIS_API_KEY:
    raise RuntimeError("DIAGNOSIS_API_KEY is not set in environment variables.")

# -------------------------------------------------
# Disease → Product Mapping (CANONICAL KEYS)
# -------------------------------------------------
DISEASE_MAP: Dict[str, Dict[str, Any]] = {

    "rice blast": {
        "products": [
            {
                "name": "Blast Force",
                "type": "Fungicide",
                "notes": "Preventive and curative fungicide formulated to control rice blast disease.",
                "dosage": "30g sachet per 15L knapsack sprayer",
            }
        ]
    },

    "brown spot (rice)": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls fungal leaf diseases including brown spot in rice.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "sheath blight": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Effective against sheath blight and related fungal infections.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "ergot": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Broad-spectrum fungicide effective against ergot in cereals.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "brown rust": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls brown rust (leaf rust) in wheat and other cereals.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "leaf blight": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls fungal leaf blight diseases.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "leaf spot": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Manages fungal leaf spot diseases.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "anthracnose": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls anthracnose in fruit trees.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "mango scab": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Effective against mango scab.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "powdery mildew": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Curative fungicide for powdery mildew.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "downy mildew": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls downy mildew under humid conditions.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "early blight": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Manages early blight in vegetables.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "late blight": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls late blight under high disease pressure.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "stem rot": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls stem and basal rot diseases.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "root rot": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Manages fungal root rot diseases.",
                "dosage": "Refer to product label",
            }
        ]
    },

    "unknown": {
        "products": []
    },
}

# -------------------------------------------------
# Disease Aliases → Canonical Keys
# -------------------------------------------------
DISEASE_ALIASES: Dict[str, str] = {
    # Rice
    "blast": "rice blast",
    "blast disease": "rice blast",
    "rice blast disease": "rice blast",
    "pyricularia": "rice blast",
    "false smut":"rice blast",
    "sheath blight of rice": "rice blast",
    "stem rust":"rice blast",

    "brown spot": "brown spot (rice)",
    "rice brown spot": "brown spot (rice)",

    # Rusts
    "leaf rust": "brown rust",
    "wheat rust": "brown rust",
    "puccinia rust": "brown rust",
    "puccinia triticina": "brown rust",

    # General
    "anthracnose disease": "anthracnose",
    "mango anthracnose": "anthracnose",
    "powdery mildew disease": "powdery mildew",
    "downy mildew disease": "downy mildew",
    "sheath blight disease": "sheath blight",

    "unknown_disease": "unknown",
}

# -------------------------------------------------
# Optional LLM
# -------------------------------------------------
llm_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY and OpenAI else None

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Crop Disease Cure App (MVP)")
BASE_DIR = Path(__file__).resolve().parent

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Models
# -------------------------------------------------
class ProductRec(BaseModel):
    name: str
    type: Optional[str] = None
    notes: Optional[str] = None
    dosage: Optional[str] = None


class DiagnoseCureResponse(BaseModel):
    disease_name: str
    probability: float
    crop_name: Optional[str] = None
    crop_probability: Optional[float] = None
    recommended_products: List[ProductRec]
    raw_api_response: Dict[str, Any]


class ExplainRequest(BaseModel):
    disease_name: str
    crop_name: Optional[str] = None


class ExplainResponse(BaseModel):
    answer: str


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_key(value: str) -> str:
    return (value or "").strip().lower()


def resolve_disease_name(raw_name: str) -> str:
    key = normalize_key(raw_name)
    return DISEASE_ALIASES.get(key, key)


def best_suggestion(items: List[dict]) -> Optional[dict]:
    return max(items, key=lambda x: x.get("probability", 0)) if items else None


def lookup_products(disease_name: str) -> List[ProductRec]:
    canonical = resolve_disease_name(disease_name)

    if canonical in DISEASE_MAP:
        return [
            ProductRec(**p)
            for p in DISEASE_MAP[canonical]["products"]
        ]

    return []


def call_crop_api(image_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    resp = requests.post(
        DIAGNOSIS_CREATE_URL,
        headers={
            "Api-Key": DIAGNOSIS_API_KEY,
            "accept": "application/json",
        },
        files=[("images", (filename, image_bytes, content_type or "image/jpeg"))],
        data={"similar_images": "true"},
        timeout=90,
    )

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose-cure", response_model=DiagnoseCureResponse)
async def diagnose_cure(file: UploadFile = File(...)):
    raw = call_crop_api(
        image_bytes=await file.read(),
        filename=file.filename or "upload.jpg",
        content_type=file.content_type or "image/jpeg",
    )

    disease_name = "unknown"
    disease_prob = 0.0
    crop_name = None
    crop_prob = None

    try:
        best_disease = best_suggestion(raw["result"]["disease"]["suggestions"])
        if best_disease:
            disease_name = best_disease.get("name", "unknown")
            disease_prob = float(best_disease.get("probability", 0))
    except Exception:
        pass

    try:
        best_crop = best_suggestion(raw["result"]["crop"]["suggestions"])
        if best_crop:
            crop_name = best_crop.get("name")
            crop_prob = float(best_crop.get("probability", 0))
    except Exception:
        pass

    products = lookup_products(disease_name)

    return DiagnoseCureResponse(
        disease_name=disease_name,
        probability=disease_prob,
        crop_name=crop_name,
        crop_probability=crop_prob,
        recommended_products=products,
        raw_api_response=raw,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if not llm_client:
        raise HTTPException(400, "OPENAI_API_KEY not configured")

    products = lookup_products(req.disease_name)

    product_text = "\n".join(
        f"- {p.name}: {p.notes or ''} {p.dosage or ''}".strip()
        for p in products
    ) or "- No mapped products found."

    prompt = f"""
Explain this disease to a farmer:

Disease: {req.disease_name}
Crop: {req.crop_name or "unknown"}

Recommended products:
{product_text}

Explain:
- What it is
- What to do
- How products help
- Safety reminder
"""

    completion = llm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return ExplainResponse(answer=completion.choices[0].message.content.strip())
