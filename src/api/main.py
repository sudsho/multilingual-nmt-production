"""FastAPI app for batch translation. Skeleton."""

from fastapi import FastAPI

app = FastAPI(title="multilingual-nmt-production")


@app.get("/health")
def health():
    return {"status": "ok"}
