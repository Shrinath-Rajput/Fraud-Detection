from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd

from src.Pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
    V9: float = Form(0),
    V10: float = Form(0),
    V11: float = Form(0),
    V12: float = Form(0),
    V13: float = Form(0),
    V14: float = Form(0),
    V15: float = Form(0),
    V16: float = Form(0),
    V17: float = Form(0),
    V18: float = Form(0),
    V19: float = Form(0),
    V20: float = Form(0),
    V21: float = Form(0),
    V22: float = Form(0),
    V23: float = Form(0),
    V24: float = Form(0),
    V25: float = Form(0),
    V26: float = Form(0),
    V27: float = Form(0),
    V28: float = Form(0),
    Amount: float = Form(0)
):

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
        "V9":[V9],
        "V10":[V10],
        "V11":[V11],
        "V12":[V12],
        "V13":[V13],
        "V14":[V14],
        "V15":[V15],
        "V16":[V16],
        "V17":[V17],
        "V18":[V18],
        "V19":[V19],
        "V20":[V20],
        "V21":[V21],
        "V22":[V22],
        "V23":[V23],
        "V24":[V24],
        "V25":[V25],
        "V26":[V26],
        "V27":[V27],
        "V28":[V28],
        "Amount":[Amount]
    }

    df = pd.DataFrame(data)

    pipeline = PredictPipeline()

    prediction = pipeline.predict(df)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": int(prediction[0])}
    )