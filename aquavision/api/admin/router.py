import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from aquavision.api.admin.engine import DynamicAnalyticsEngine
from aquavision.api.admin.views import get_dashboard_html

admin_router = APIRouter(prefix="/admin", tags=["Research Dashboard"])
engine = DynamicAnalyticsEngine()

@admin_router.get("/dashboard", response_class=HTMLResponse)
def render_research_dashboard():
    # No longer injecting a string; letting the frontend fetch the data securely
    return get_dashboard_html()

@admin_router.get("/api/metrics")
def get_metrics_api():
    # New endpoint serving raw JSON data
    return {
        "dataset_metrics": engine.get_dataset_metrics(),
        "model_metrics": engine.get_model_metrics()
    }