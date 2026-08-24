import gradio as gr
import numpy as np
from config import load_config
from aquavision.predict import AquaPredictor

cfg = load_config()
predictor = AquaPredictor(cfg, dataset_type="fish")

def analyze_aquaculture(image, ph, temp, do, salinity, ammonia, turbidity):
    if image is None:
        return "Please upload an image.", {}

    predictions, top_pred, confidence = predictor.predict(
        image, ph, temp, do, salinity, ammonia, turbidity
    )
    
    # Generate action status recommendation based on water quality checks
    alerts = []
    if ph < 6.5 or ph > 8.5:
        alerts.append(f"pH level ({ph}) is outside ideal range (6.5 - 8.5).")
    if do < 5.0:
        alerts.append(f"Low Dissolved Oxygen ({do} mg/L). Turn on aeration.")
    if ammonia > 0.05:
        alerts.append(f"High Ammonia ({ammonia} mg/L). Perform water exchange.")

    status_msg = f"### Diagnosis: **{top_pred.upper()}** (Confidence: {confidence:.2%})\n\n"
    if alerts:
        status_msg += "**Water Quality Warnings:**\n" + "\n".join([f"- {a}" for a in alerts])
    else:
        status_msg += "Water quality parameters are within nominal operational ranges."

    return status_msg, predictions

demo = gr.Interface(
    fn=analyze_aquaculture,
    inputs=[
        gr.Image(label="Fish Sample Image"),
        gr.Slider(4.0, 10.0, value=7.2, step=0.1, label="pH Level"),
        gr.Slider(15.0, 40.0, value=26.0, step=0.5, label="Temperature (°C)"),
        gr.Slider(0.5, 12.0, value=6.5, step=0.1, label="Dissolved Oxygen (mg/L)"),
        gr.Slider(0.0, 40.0, value=15.0, step=0.5, label="Salinity (ppt)"),
        gr.Slider(0.0, 3.0, value=0.02, step=0.01, label="Ammonia (mg/L)"),
        gr.Slider(1.0, 100.0, value=12.0, step=1.0, label="Turbidity (NTU)"),
    ],
    outputs=[
        gr.Markdown(label="Diagnostic Results"),
        gr.Label(num_top_classes=5, label="Class Probabilities"),
    ],
    title="AquaVision: AI-Powered Multimodal Aquaculture Diagnostics",
    description="Upload a fish sample image and input real-time water parameters for joint vision-tabular disease classification.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
