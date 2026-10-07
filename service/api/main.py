from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="Object Recognition API", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(_request: PredictRequest) -> PredictResponse:
    raise HTTPException(status_code=501, detail="API wiring is scaffolded; CLI pipeline integration is the next step.")
