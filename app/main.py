from fastapi import FastAPI

from app.predict import predict_price
from app.schemas import HouseInput


app = FastAPI(
    title="House Prices Prediction API",
    description="Kaggle House Prices model inference API",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: HouseInput):
    predicted_price = predict_price(input_data.model_dump())

    return {
        "predicted_price": round(predicted_price, 2)
    }