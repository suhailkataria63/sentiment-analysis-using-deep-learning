# src/api/eda_reports_routes.py

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["EDA"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# ✅ Your actual filenames from reports/figures/
STATIC_PLOTS = {
    "class_distribution": "01_class_distribution.png",
    "text_length_distribution": "02_text_length_distribution.png",
    "top_tokens_by_sentiment": "03_top_tokens_by_sentiment.png",
}

@router.get("/eda/static")
def list_static_plots():
    return {"available": sorted(STATIC_PLOTS.keys()), "dir": str(FIG_DIR)}

@router.get("/eda/static/{plot_id}")
def get_static_plot(plot_id: str):
    if plot_id not in STATIC_PLOTS:
        raise HTTPException(status_code=400, detail=f"Unknown plot_id: {plot_id}")

    path = FIG_DIR / STATIC_PLOTS[plot_id]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    return FileResponse(path, media_type="image/png")

@router.get("/eda/ping")
def eda_ping():
    return {"eda": "pong"}
