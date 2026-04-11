from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os

from src.Pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# ✅ CORRECT
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    try:
        df = pd.DataFrame({
            "Time":[1000],
            "V1":[0], "V2":[0], "V3":[0], "V4":[0], "V5":[0],
            "V6":[0], "V7":[0], "V8":[0], "V9":[0], "V10":[0],
            "V11":[0], "V12":[0], "V13":[0], "V14":[0],
            "V15":[0], "V16":[0], "V17":[0], "V18":[0],
            "V19":[0], "V20":[0], "V21":[0], "V22":[0],
            "V23":[0], "V24":[0], "V25":[0], "V26":[0],
            "V27":[0], "V28":[0],
            "Amount":[100]
        })

        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        return templates.TemplateResponse(
            "result.html",
            {"request": request, "prediction": int(prediction[0])}
        )

    except Exception as e:
        return HTMLResponse(f"<h2>ERROR: {str(e)}</h2>")