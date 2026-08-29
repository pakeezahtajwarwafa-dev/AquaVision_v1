def get_dashboard_html() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AquaVision BD — Computer Vision Research Hub</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script>
            tailwind.config = { darkMode: 'class', theme: { extend: { colors: { brand: '#0ea5e9', darkbg: '#0f172a', cardbg: '#1e293b' } } } }
        </script>
    </head>
    <body class="bg-darkbg text-slate-200 font-sans min-h-screen p-4 md:p-8">
        <div class="max-w-7xl mx-auto space-y-8">

            <div class="border-b border-slate-700 pb-6">
                <h1 class="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-brand to-emerald-400 tracking-tight">
                    AquaVision Image Analytics Hub
                </h1>
                <p class="text-slate-400 mt-2 font-medium">Dynamic Dataset Detection & Model Artifact Tracking</p>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Total Datasets</p>
                    <h2 id="kpi-datasets" class="text-4xl font-black text-white mt-2">0</h2>
                </div>
                <div class="bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Total Image Samples</p>
                    <h2 id="kpi-total" class="text-4xl font-black text-brand mt-2">0</h2>
                </div>
                <div class="bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Zero-Leakage Held-out</p>
                    <h2 id="kpi-heldout" class="text-4xl font-black text-purple-400 mt-2">0</h2>
                </div>
                <div class="bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                    <p class="text-xs font-bold text-slate-400 uppercase tracking-wider">Architectures Tracked</p>
                    <h2 id="kpi-models" class="text-4xl font-black text-emerald-400 mt-2">0</h2>
                </div>
            </div>

            <div id="dynamic-datasets"></div>

            <div class="bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                <h3 class="text-xl font-bold text-white mb-6"><i class="fas fa-microchip text-emerald-400 mr-2"></i>Compiled Model Artifacts</h3>
                <div id="dynamic-models" class="overflow-x-auto"></div>
            </div>

        </div>

        <script>
            async function fetchMetrics() {
                try {
                    const response = await fetch('/admin/api/metrics');
                    const DATA = await response.json();

                    document.getElementById('kpi-datasets').innerText = DATA.dataset_metrics.global_stats.dataset_count;
                    document.getElementById('kpi-total').innerText = DATA.dataset_metrics.global_stats.total_samples.toLocaleString();
                    document.getElementById('kpi-heldout').innerText = DATA.dataset_metrics.global_stats.total_heldout.toLocaleString();

                    const modelsCount = Object.keys(DATA.model_metrics.pytorch_checkpoints).length;
                    document.getElementById('kpi-models').innerText = modelsCount;

                    const container = document.getElementById('dynamic-datasets');
                    let html = '';

                    for (const [dsName, stats] of Object.entries(DATA.dataset_metrics.datasets)) {
                        const title = dsName.charAt(0).toUpperCase() + dsName.slice(1);
                        html += `
                        <div class="mb-8 bg-cardbg p-6 rounded-2xl border border-slate-700 shadow-xl">
                            <div class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
                                <h3 class="text-2xl font-bold text-white"><i class="fas fa-folder-open text-brand mr-2"></i>Dataset: ${title}</h3>
                                <span class="px-3 py-1 bg-slate-800 rounded-lg text-sm font-bold text-slate-300">Total Samples: ${stats.total.toLocaleString()}</span>
                            </div>
                            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                <div>
                                    <h4 class="text-sm font-bold text-slate-400 mb-4 text-center">Class Distribution</h4>
                                    <div class="h-64"><canvas id="classChart-${dsName}"></canvas></div>
                                </div>
                                <div>
                                    <h4 class="text-sm font-bold text-slate-400 mb-4 text-center">Stratified Folds (Orig vs Aug)</h4>
                                    <div class="h-64"><canvas id="foldChart-${dsName}"></canvas></div>
                                </div>
                            </div>
                        </div>`;
                    }
                    container.innerHTML = html;

                    for (const [dsName, stats] of Object.entries(DATA.dataset_metrics.datasets)) {
                        new Chart(document.getElementById(`classChart-${dsName}`).getContext('2d'), {
                            type: 'doughnut',
                            data: {
                                labels: Object.keys(stats.class_balance),
                                datasets: [{ data: Object.values(stats.class_balance), backgroundColor: ['#0ea5e9','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#14b8a6','#f97316'], borderWidth: 0 }]
                            },
                            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#cbd5e1' } } }, cutout: '70%' }
                        });

                        const foldLabels = Object.keys(stats.fold_breakdown);
                        const originals = foldLabels.map(l => stats.fold_breakdown[l].original);
                        const augs = foldLabels.map(l => stats.fold_breakdown[l].augmented);

                        new Chart(document.getElementById(`foldChart-${dsName}`).getContext('2d'), {
                            type: 'bar',
                            data: {
                                labels: foldLabels,
                                datasets: [
                                    { label: 'Original', data: originals, backgroundColor: '#0ea5e9' },
                                    { label: 'Augmented', data: augs, backgroundColor: '#8b5cf6' }
                                ]
                            },
                            options: { responsive: true, maintainAspectRatio: false, scales: { x: { stacked: true }, y: { stacked: true } }, plugins: { legend: { labels: { color: '#cbd5e1' } } } }
                        });
                    }

                    const modelContainer = document.getElementById('dynamic-models');
                    const ptModels = DATA.model_metrics.pytorch_checkpoints;
                    const onnxModels = DATA.model_metrics.onnx_exports;

                    let modelHtml = '<table class="w-full text-left border-collapse"><thead class="border-b border-slate-700"><tr><th class="py-3 px-4 text-slate-400 uppercase text-xs">Model Architecture</th><th class="py-3 px-4 text-slate-400 uppercase text-xs">PyTorch Assets (.pt)</th><th class="py-3 px-4 text-slate-400 uppercase text-xs">Edge Assets (.onnx)</th></tr></thead><tbody>';

                    const allArchs = new Set([...Object.keys(ptModels), ...Object.keys(onnxModels)]);
                    for (const arch of allArchs) {
                        const ptCount = ptModels[arch] ? ptModels[arch].length : 0;
                        const onnxCount = onnxModels[arch] ? onnxModels[arch].length : 0;
                        modelHtml += `<tr class="border-b border-slate-800 hover:bg-slate-800/50">
                            <td class="py-4 px-4 font-bold text-brand uppercase">${arch}</td>
                            <td class="py-4 px-4"><span class="px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded">${ptCount} Checkpoints</span></td>
                            <td class="py-4 px-4"><span class="px-3 py-1 ${onnxCount > 0 ? 'bg-orange-500/10 text-orange-400 border border-orange-500/20' : 'bg-slate-800 text-slate-500'} rounded">${onnxCount} Exports</span></td>
                        </tr>`;
                    }
                    modelHtml += '</tbody></table>';

                    if(allArchs.size === 0) {
                        modelContainer.innerHTML = '<p class="text-slate-500 italic text-center py-6">No model architectures detected in /checkpoints.</p>';
                    } else {
                        modelContainer.innerHTML = modelHtml;
                    }
                } catch (error) {
                    console.error("Error loading dashboard data:", error);
                }
            }
            window.onload = fetchMetrics;
        </script>
    </body>
    </html>
    """