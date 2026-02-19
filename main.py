from fastapi import FastAPI
from inference.api import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Vector Store Builder")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "API is running!"}


# @app.on_event("startup")
# async def startup_event():
#     preload_resources(languages=["en", "bn"])
