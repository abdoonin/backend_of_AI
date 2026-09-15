import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./medical_ai.db")
os.environ.setdefault("ENVIRONMENT", "production")

import gradio as gr
import uvicorn
from main import app as fastapi_app

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

# Mount Gradio onto the FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
