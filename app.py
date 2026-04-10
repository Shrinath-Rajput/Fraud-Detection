from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import sys

from src.Pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

# 🔥 Base path fix (Render compatible)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Templates path
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# =========================
# HOME PAGE
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================
# PREDICT ROUTE
# =========================
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    Time: float = Form(0),
    V1: float = Form(0),
    V2: float = Form(0),
    V3: float = Form(0),
    V4: float = Form(0),
    V5: float = Form(0),
    V6: float = Form(0),
    V7: float = Form(0),
    V8: float = Form(0),
    Amount: float = Form(0)
):
    try:
        # 🔥 Input Data
        data = {
            "Time": [Time],
            "V1": [V1],
            "V2": [V2],
            "V3": [V3],
            "V4": [V4],
            "V5": [V5],
            "V6": [V6],
            "V7": [V7],
            "V8": [V8],
            "Amount": [Amount]
        }

        df = pd.DataFrame(data)

        # 🔥 Prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        result = int(prediction[0])

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": result
            }
        )

    except Exception as e:
        return HTMLResponse(f"<h2>Error: {str(e)}</h2>")