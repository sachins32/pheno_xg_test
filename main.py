from typing import Sequence
from fastapi import FastAPI
from fastapi.logger import logger
import numpy as np
from . import pheno_xgb_classifier as ph


app = FastAPI()

model = ph.pheno_classifier()


@app.get("/")
def read_root():
    return "OK"


@app.post("/predict")
def predict(input: Sequence[float]):
    print(f"Input: {input}")
    x = np.array([input])
    result = model.predict(x)
    print(f"Result: {result}")
    return result.tolist()