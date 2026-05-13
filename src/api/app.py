from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List

from src.dl.predictor import predict, predict_batch
from src.api.eda_reports_routes import router as eda_reports_router

# ✅ Create ONE FastAPI app (do NOT redefine later)
app = FastAPI(
    title="Sentiment API (GRU)",
    version="1.2",
)

# ✅ Include routers AFTER app is created
app.include_router(eda_reports_router)

templates = Jinja2Templates(directory="src/api/templates")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to classify")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_api(req: PredictRequest, max_len: int = 50, threshold: float = 0.5):
    result = predict(text=req.text, max_len=max_len, threshold=threshold)
    return {"text": req.text, **result}


@app.post("/predict_batch")
def predict_batch_api(req: PredictBatchRequest, max_len: int = 50, threshold: float = 0.5):
    texts = [t for t in req.texts if isinstance(t, str) and t.strip() != ""]
    results = predict_batch(texts=texts, max_len=max_len, threshold=threshold)
    return {"count": len(results), "results": results}
