from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import dl, eda, ml, predict

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

app.include_router(eda.router, tags=["EDA"])
app.include_router(ml.router, prefix="/ml", tags=["ML"])
app.include_router(dl.router, prefix="/dl", tags=["DL"])
app.include_router(predict.router, prefix="/predict", tags=["Predict"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
