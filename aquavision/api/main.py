import json
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from aquavision.api.admin.router import admin_router
from aquavision.engine.predictor import AquaPredictor

app = FastAPI(
    title="AquaVision BD Diagnostic API Engine",
    description="Multimodal Aquatic Disease Diagnostic API and Research Hub",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

# In-memory cache to prevent reloading heavy PyTorch weights on every API call
_model_cache = {}

def get_cached_predictor(species: str):
    if species not in _model_cache:
        try:
            _model_cache[species] = AquaPredictor(species=species, fold=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    return _model_cache[species]

@app.get("/")
def redirect_to_dashboard():
    return RedirectResponse(url="/admin/dashboard")

@app.get("/health")
def health_check():
    return {"status": "online", "system": "AquaVision Engine v2"}

@app.post("/predict-multimodal")
async def process_multimodal_prediction(
    file: UploadFile = File(...),
    species: str = Form("fish"),
    water_quality: str = Form("{}")
):
    try:
        wq_dict = json.loads(water_quality)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        predictor = get_cached_predictor(species)
        results = predictor.predict(img_rgb, wq_dict)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test", tags=["Testing UI"])
def serve_rich_testing_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AquaVision Multimodal Engine</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #38bdf8; --border: #334155; --success: #10b981; --error: #ef4444; }
            body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; margin: 0; }
            .container { max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
            .card { background: var(--card); border: 1px solid var(--border); border-radius: 1rem; padding: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }
            h1, h2 { color: var(--accent); margin-top: 0; }
            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
            label { display: block; font-size: 0.875rem; color: #94a3b8; margin-bottom: 0.5rem; }
            input, select { width: 100%; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid var(--border); background: var(--bg); color: var(--text); box-sizing: border-box; }
            button { width: 100%; padding: 1rem; background: var(--accent); color: #000; border: none; border-radius: 0.5rem; font-weight: bold; font-size: 1rem; cursor: pointer; margin-top: 1rem; }
            button:hover { opacity: 0.9; }
            #preview { max-width: 100%; border-radius: 0.5rem; display: none; margin-top: 1rem; border: 1px solid var(--border); }
            
            /* Results Panel Styling */
            .main-result { background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent); padding: 1.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 2rem; display: none; }
            .main-result h2 { margin: 0; font-size: 2rem; color: var(--text); }
            .main-result span { color: var(--accent); font-weight: bold; font-size: 1.25rem; }
            
            /* Warning Box Styling */
            .mismatch-alert { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--error); padding: 1.5rem; border-radius: 0.5rem; color: #fca5a5; margin-bottom: 1rem; display: none; }
            .mismatch-alert h3 { margin: 0 0 0.5rem 0; color: var(--error); }
            
            .prob-item { margin-bottom: 1rem; }
            .prob-header { display: flex; justify-content: space-between; font-size: 0.875rem; margin-bottom: 0.25rem; }
            .prob-track { background: var(--bg); height: 8px; border-radius: 4px; overflow: hidden; }
            .prob-fill { background: var(--success); height: 100%; width: 0%; transition: width 0.6s ease-out; }
            #error-box { color: #ef4444; margin-top: 1rem; display: none; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; margin-bottom: 2rem;">🔬 AquaVision Multimodal Diagnostic Hub</h1>
        <div class="container">
            
            <!-- Left Panel: Data Inputs -->
            <div class="card">
                <h2>1. Input Parameters</h2>
                <form id="engineForm">
                    <label>Target Species</label>
                    <select id="species" style="margin-bottom: 1rem;">
                        <option value="fish">Fish</option>
                        <option value="shrimp">Shrimp</option>
                    </select>
                    
                    <h3 style="color: #94a3b8; font-size: 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">Water Quality Data</h3>
                    <div class="form-grid">
                        <div><label>Temperature (°C)</label><input type="number" id="temp" step="0.1" value="25.0"></div>
                        <div><label>pH Level</label><input type="number" id="ph" step="0.1" value="7.5"></div>
                        <div><label>Dissolved Oxygen (mg/L)</label><input type="number" id="do" step="0.1" value="6.0"></div>
                        <div><label>Ammonia (mg/L)</label><input type="number" id="nh3" step="0.01" value="0.01"></div>
                    </div>

                    <h3 style="color: #94a3b8; font-size: 1rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; margin-top: 1.5rem;">Visual Subject</h3>
                    <input type="file" id="file" accept="image/*" required onchange="previewImage(event)">
                    <img id="preview" alt="Image preview">

                    <button type="submit" id="submitBtn">Run Diagnostic Inference</button>
                    <div id="error-box"></div>
                </form>
            </div>

            <!-- Right Panel: Inference Results -->
            <div class="card">
                <h2>2. Diagnostic Report</h2>
                <p id="placeholder" style="color: #94a3b8;">Awaiting data input...</p>
                
                <!-- Species Mismatch Alert Banner -->
                <div class="mismatch-alert" id="mismatchAlertBox">
                    <h3>🚨 Species Mismatch Detected</h3>
                    <p id="mismatchMessage" style="margin: 0;"></p>
                </div>

                <div class="main-result" id="mainResultBox">
                    <h2 id="topPrediction">--</h2>
                    <span id="topConfidence">--% Confidence</span>
                </div>

                <div id="probBreakdown"></div>
            </div>
        </div>

        <script>
            function previewImage(event) {
                const reader = new FileReader();
                reader.onload = function(){
                    const output = document.getElementById('preview');
                    output.src = reader.result;
                    output.style.display = 'block';
                };
                reader.readAsDataURL(event.target.files[0]);
            }

            document.getElementById('engineForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                const errBox = document.getElementById('error-box');
                const mainResultBox = document.getElementById('mainResultBox');
                const mismatchBox = document.getElementById('mismatchAlertBox');
                const breakdown = document.getElementById('probBreakdown');

                btn.textContent = 'Running Inference...';
                btn.disabled = true;
                errBox.style.display = 'none';
                mainResultBox.style.display = 'none';
                mismatchBox.style.display = 'none';
                breakdown.innerHTML = '';

                const formData = new FormData();
                formData.append('file', document.getElementById('file').files[0]);
                formData.append('species', document.getElementById('species').value);
                
                const wq = {
                    temperature: parseFloat(document.getElementById('temp').value),
                    ph: parseFloat(document.getElementById('ph').value),
                    dissolved_oxygen: parseFloat(document.getElementById('do').value),
                    ammonia: parseFloat(document.getElementById('nh3').value)
                };
                formData.append('water_quality', JSON.stringify(wq));

                try {
                    const res = await fetch('/predict-multimodal', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    if (res.ok) {
                        document.getElementById('placeholder').style.display = 'none';
                        
                        if (data.is_mismatch) {
                            // Show Red Alert Box for Species Mismatch
                            document.getElementById('mismatchMessage').textContent = data.message;
                            mismatchBox.style.display = 'block';
                        } else {
                            // Show Normal Diagnosis Results
                            mainResultBox.style.display = 'block';
                            document.getElementById('topPrediction').textContent = data.prediction;
                            document.getElementById('topConfidence').textContent = (data.confidence * 100).toFixed(1) + '% Confidence';

                            breakdown.innerHTML = '<h3 style="margin-bottom: 1rem;">Probability Breakdown</h3>';
                            const sortedProbs = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
                            
                            sortedProbs.forEach(([cls, prob]) => {
                                const pct = (prob * 100).toFixed(1);
                                breakdown.innerHTML += `
                                    <div class="prob-item">
                                        <div class="prob-header"><span>${cls}</span><span>${pct}%</span></div>
                                        <div class="prob-track"><div class="prob-fill" style="width: ${pct}%"></div></div>
                                    </div>
                                `;
                            });
                        }
                    } else {
                        errBox.textContent = data.detail || 'Prediction failed.';
                        errBox.style.display = 'block';
                    }
                } catch (err) {
                    errBox.textContent = 'Network or server error.';
                    errBox.style.display = 'block';
                } finally {
                    btn.textContent = 'Run Diagnostic Inference';
                    btn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
