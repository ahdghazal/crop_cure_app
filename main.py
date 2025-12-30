import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
# Disease → Product Mapping (INLINE, NO FILE)
# -------------------------------------------------
DISEASE_MAP: Dict[str, Dict[str, Any]] = {
    "rice blast": {
        "products": [
            {
                "name": "Blast Force",
                "type": "Fungicide",
                "notes": "Preventive and curative fungicide for rice blast disease.",
                "dosage": "30g sachet per 15L knapsack sprayer",
            }
        ]
    },
    "ergot": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Broad-spectrum fungicide effective against fungal diseases.",
                "dosage": "Refer to product label",
            }
        ]
    },
    "anthracnose": {
        "products": [
            {
                "name": "Fungi Care",
                "type": "Fungicide",
                "notes": "Controls anthracnose and related fungal infections.",
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
}

# -------------------------------------------------
# Optional LLM client
# -------------------------------------------------
llm_client = None
if OPENAI_API_KEY and OpenAI is not None:
    llm_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Crop Disease Cure App (MVP)")
BASE_DIR = Path(__file__).resolve().parent

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = BASE_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")

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


def best_suggestion(items: List[dict]) -> Optional[dict]:
    if not items:
        return None
    return max(items, key=lambda x: x.get("probability", 0))


def lookup_products(disease_name: str) -> List[ProductRec]:
    key = normalize_key(disease_name)

    if key in DISEASE_MAP:
        return [ProductRec(**p) for p in DISEASE_MAP[key]["products"]]

    # simple aliases
    aliases = {
        "blast": "rice blast",
        "rice blast disease": "rice blast",
    }

    if key in aliases and aliases[key] in DISEASE_MAP:
        return [ProductRec(**p) for p in DISEASE_MAP[aliases[key]]["products"]]

    return []


def call_crop_api(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> Dict[str, Any]:
    headers = {
        "Api-Key": DIAGNOSIS_API_KEY,
        "accept": "application/json",
    }

    files = [
        ("images", (filename, image_bytes, content_type or "image/jpeg"))
    ]

    data = {
        "similar_images": "true"
    }

    resp = requests.post(
        DIAGNOSIS_CREATE_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=90,
    )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail={"error": resp.text},
        )

    return resp.json()


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose-cure", response_model=DiagnoseCureResponse)
async def diagnose_cure(file: UploadFile = File(...)):
    image_bytes = await file.read()

    raw = call_crop_api(
        image_bytes=image_bytes,
        filename=file.filename or "upload.jpg",
        content_type=file.content_type or "image/jpeg",
    )

    disease_name = "unknown"
    disease_prob = 0.0
    crop_name = None
    crop_prob = None

    try:
        disease_suggestions = raw["result"]["disease"]["suggestions"]
        best_disease = best_suggestion(disease_suggestions)
        if best_disease:
            disease_name = best_disease.get("name", "unknown")
            disease_prob = float(best_disease.get("probability", 0.0))
    except Exception:
        pass

    try:
        crop_suggestions = raw["result"]["crop"]["suggestions"]
        best_crop = best_suggestion(crop_suggestions)
        if best_crop:
            crop_name = best_crop.get("name")
            crop_prob = float(best_crop.get("probability", 0.0))
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
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY not configured on server.",
        )

    products = lookup_products(req.disease_name)
    product_text = (
        "\n".join(
            f"- {p.name}: {p.notes or ''} {p.dosage or ''}".strip()
            for p in products
        )
        if products
        else "- No mapped products found."
    )

    prompt = f"""
You are an agricultural advisor.

Disease: {req.disease_name}
Crop: {req.crop_name or "unknown"}

Recommended products:
{product_text}

Explain clearly and simply:
- What the disease is
- What farmers should do
- How products help
- Safety reminder
"""

    completion = llm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return ExplainResponse(
        answer=completion.choices[0].message.content.strip()
    )
