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
# PREDICT
# =========================
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Time: float = Form(...),
    V1: float = Form(...),
    V2: float = Form(...),
    V3: float = Form(...),
    V4: float = Form(...),
    V5: float = Form(...),
    V6: float = Form(...),
    V7: float = Form(...),
    V8: float = Form(...),
    Amount: float = Form(...)
):
    try:
        data = {
            "Time":[Time],
            "V1":[V1],
            "V2":[V2],
            "V3":[V3],
            "V4":[V4],
            "V5":[V5],
            "V6":[V6],
            "V7":[V7],
            "V8":[V8],

            "V9":[0], "V10":[0], "V11":[0], "V12":[0],
            "V13":[0], "V14":[0], "V15":[0], "V16":[0],
            "V17":[0], "V18":[0], "V19":[0], "V20":[0],
            "V21":[0], "V22":[0], "V23":[0], "V24":[0],
            "V25":[0], "V26":[0], "V27":[0], "V28":[0],

            "Amount":[Amount]
        }

        df = pd.DataFrame(data)

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