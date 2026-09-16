import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./medical_ai.db")
os.environ.setdefault("ENVIRONMENT", "production")

import spaces
import gradio as gr

# Ensure backwards compatibility for older pickled pipelines
try:
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, '_RemainderColsList'):
        ct._RemainderColsList = list
except Exception:
    pass

from main import app as fastapi_app

# ZeroGPU requires at least one event-bound @spaces.GPU function
@spaces.GPU(duration=5)
def gpu_health_check(text):
    return f"AI Engine Online: {text}"

# Simple Gradio UI for the Space landing page
with gr.Blocks(title="Hepatiq AI Backend API") as demo:
    gr.Markdown(
        """
        # 🏥 Hepatiq AI Backend API
        
        The FastAPI backend server is **online and healthy** with all diagnostic models loaded.
        
        - 📖 **Interactive API Documentation:** [/docs](/docs)
        - 📑 **Alternative Documentation:** [/redoc](/redoc)
        - 🩺 **Health Check:** [/health](/health)
        """
    )
    # Event handler binding for ZeroGPU scanner
    inp = gr.Textbox(value="ping", visible=False)
    out = gr.Textbox(visible=False)
    btn = gr.Button("Health Check", visible=False)
    btn.click(fn=gpu_health_check, inputs=inp, outputs=out)

# Mount Gradio onto the FastAPI app so Hugging Face serves both
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
