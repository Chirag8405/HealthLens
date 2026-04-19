import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import dispose_engine
from api.routers import dl, eda, health, ml, predict, predictions

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Safe fallback when GPU runtime has already been initialized.
            pass
else:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

app = FastAPI(
    title="Intelligent Healthcare Data Analytics API",
    version="0.1.0",
    description=(
        "Decision Support API for EDA, machine learning, deep learning, and prediction modules."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup() -> None:
    print("HealthLens API started. Models load on first request.")


@app.on_event("shutdown")
async def shutdown() -> None:
    await dispose_engine()


app.include_router(health.router, tags=["Health"])
app.include_router(eda.router, tags=["EDA"])
app.include_router(ml.router, prefix="/ml", tags=["ML"])
app.include_router(dl.router, prefix="/dl", tags=["DL"])
app.include_router(predict.router, prefix="/predict", tags=["Predict"])
app.include_router(predictions.router, tags=["Predictions"])
