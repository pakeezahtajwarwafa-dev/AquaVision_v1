from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from aquavision.api.analytics import DeveloperAnalyticsEngine

app = FastAPI(
    title="AquaVision BD — Developer Analytics Engine",
    description="Developer dashboard and API engine for multimodal aquatic diagnostic models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analytics_engine = DeveloperAnalyticsEngine()

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AquaVision BD — Developer Analytics</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body class="bg-slate-900 text-slate-100 font-sans antialiased min-h-screen p-6">
        <div class="max-w-7xl mx-auto space-y-6">
            
            <!-- Header -->
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                <div>
                    <h1 class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                        AquaVision BD
                    </h1>
                    <p class="text-slate-400 text-sm mt-1">Multimodal Dataset & System Infrastructure Analytics</p>
                </div>
                <div class="mt-4 md:mt-0 flex gap-3">
                    <a href="/docs" target="_blank" class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-xl text-sm font-semibold transition flex items-center gap-2 shadow-lg shadow-cyan-900/40">
                        <i class="fas fa-code"></i> OpenAPI Docs
                    </a>
                    <button onclick="fetchAnalytics()" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-semibold transition flex items-center gap-2 border border-slate-600">
                        <i class="fas fa-sync-alt"></i> Refresh Data
                    </button>
                </div>
            </div>

            <!-- KPI Cards -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-lg flex items-center justify-between">
                    <div>
                        <p class="text-xs uppercase font-bold text-slate-400">Total Fish Images</p>
                        <h2 id="fish-total" class="text-3xl font-black text-cyan-400 mt-1">--</h2>
                        <span id="fish-heldout" class="text-xs text-slate-400">-- Held-out Test</span>
                    </div>
                    <div class="p-3 bg-cyan-500/10 rounded-xl text-cyan-400 text-2xl"><i class="fas fa-fish"></i></div>
                </div>

                <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-lg flex items-center justify-between">
                    <div>
                        <p class="text-xs uppercase font-bold text-slate-400">Total Shrimp Images</p>
                        <h2 id="shrimp-total" class="text-3xl font-black text-blue-400 mt-1">--</h2>
                        <span id="shrimp-aug" class="text-xs text-slate-400">-- Augmented</span>
                    </div>
                    <div class="p-3 bg-blue-500/10 rounded-xl text-blue-400 text-2xl"><i class="fas fa-shrimp"></i></div>
                </div>

                <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-lg flex items-center justify-between">
                    <div>
                        <p class="text-xs uppercase font-bold text-slate-400">PyTorch Checkpoints</p>
                        <h2 id="pt-count" class="text-3xl font-black text-emerald-400 mt-1">--</h2>
                        <span class="text-xs text-slate-400">5-Fold Models Built</span>
                    </div>
                    <div class="p-3 bg-emerald-500/10 rounded-xl text-emerald-400 text-2xl"><i class="fas fa-microchip"></i></div>
                </div>

                <div class="bg-slate-800 p-5 rounded-xl border border-slate-700 shadow-lg flex items-center justify-between">
                    <div>
                        <p class="text-xs uppercase font-bold text-slate-400">BFRI Guidelines</p>
                        <h2 id="bfri-count" class="text-3xl font-black text-purple-400 mt-1">--</h2>
                        <span class="text-xs text-slate-400">RAG Protocols Loaded</span>
                    </div>
                    <div class="p-3 bg-purple-500/10 rounded-xl text-purple-400 text-2xl"><i class="fas fa-book-medical"></i></div>
                </div>
            </div>

            <!-- Charts Section -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Fish Class Distribution Chart -->
                <div class="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <h3 class="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
                        <i class="fas fa-chart-bar text-cyan-400"></i> Fish Class Distribution
                    </h3>
                    <div class="relative h-72">
                        <canvas id="fishChart"></canvas>
                    </div>
                </div>

                <!-- Shrimp Class Distribution Chart -->
                <div class="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <h3 class="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
                        <i class="fas fa-chart-pie text-blue-400"></i> Shrimp Class Distribution
                    </h3>
                    <div class="relative h-72">
                        <canvas id="shrimpChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- System Artifacts Inventory -->
            <div class="bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl">
                <h3 class="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
                    <i class="fas fa-cubes text-emerald-400"></i> System Artifacts & Deployment Status
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/50">
                        <p class="text-xs font-bold uppercase text-slate-400 mb-2">ONNX Edge Models</p>
                        <ul id="onnx-list" class="space-y-1 text-sm font-mono text-cyan-300"></ul>
                    </div>
                    <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/50">
                        <p class="text-xs font-bold uppercase text-slate-400 mb-2">Active Scalers (.pkl)</p>
                        <p id="scalers-summary" class="text-sm font-semibold text-emerald-400"></p>
                    </div>
                    <div class="bg-slate-900/60 p-4 rounded-xl border border-slate-700/50">
                        <p class="text-xs font-bold uppercase text-slate-400 mb-2">Cross-Validation Pool</p>
                        <p id="cv-summary" class="text-sm font-semibold text-blue-400"></p>
                    </div>
                </div>
            </div>

        </div>

        <script>
            let fishChartInst = null;
            let shrimpChartInst = null;

            async function fetchAnalytics() {
                try {
                    const response = await fetch('/analytics/overview');
                    const data = await response.json();
                    
                    const fish = data.dataset_summary.fish;
                    const shrimp = data.dataset_summary.shrimp;
                    const artifacts = data.system_artifacts;

                    // Update KPI Cards
                    document.getElementById('fish-total').innerText = fish.total_images.toLocaleString();
                    document.getElementById('fish-heldout').innerText = fish.heldout_test_size + ' Held-out Test Samples';
                    
                    document.getElementById('shrimp-total').innerText = shrimp.total_images.toLocaleString();
                    document.getElementById('shrimp-aug').innerText = shrimp.augmented_samples + ' Augmented Samples';

                    document.getElementById('pt-count').innerText = artifacts.pytorch_checkpoints.length;
                    document.getElementById('bfri-count').innerText = artifacts.bfri_protocols_count;

                    // Render Fish Chart
                    renderFishChart(fish.class_distribution);

                    // Render Shrimp Chart
                    renderShrimpChart(shrimp.class_distribution);

                    // Render Artifacts List
                    const onnxList = document.getElementById('onnx-list');
                    onnxList.innerHTML = artifacts.onnx_models.map(m => <li><i class="fas fa-check-circle text-emerald-400 text-xs mr-2"></i> + m + </li>).join('');

                    document.getElementById('scalers-summary').innerText = artifacts.scaler_files.length + ' StandardScalers Configured';
                    document.getElementById('cv-summary').innerText = 'Fish: 5 Folds | Shrimp: 5 Folds';

                } catch (err) {
                    console.error("Failed to load analytics:", err);
                }
            }

            function renderFishChart(dist) {
                const ctx = document.getElementById('fishChart').getContext('2d');
                if (fishChartInst) fishChartInst.destroy();

                fishChartInst = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(dist),
                        datasets: [{
                            label: 'Images per Class',
                            data: Object.values(dist),
                            backgroundColor: 'rgba(6, 182, 212, 0.7)',
                            borderColor: '#06b6d4',
                            borderWidth: 1.5,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                            x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } }
                        }
                    }
                });
            }

            function renderShrimpChart(dist) {
                const ctx = document.getElementById('shrimpChart').getContext('2d');
                if (shrimpChartInst) shrimpChartInst.destroy();

                shrimpChartInst = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: Object.keys(dist),
                        datasets: [{
                            data: Object.values(dist),
                            backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
                            borderWidth: 2,
                            borderColor: '#1e293b'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'right', labels: { color: '#94a3b8', boxWidth: 12 } }
                        }
                    }
                });
            }

            window.onload = fetchAnalytics;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    return {"status": "online", "mode": "developer_analytics"}

@app.get("/analytics/overview")
def get_full_overview():
    return analytics_engine.get_system_summary()

@app.get("/analytics/manifest/{species}")
def get_species_manifest_analytics(species: str):
    species = species.lower()
    if species not in ["fish", "shrimp"]:
        raise HTTPException(status_code=400, detail="Invalid species. Must be 'fish' or 'shrimp'.")
    
    stats = analytics_engine.get_manifest_stats(species)
    if stats.get("status") == "missing":
        raise HTTPException(status_code=444, detail=stats["message"])
    return stats
