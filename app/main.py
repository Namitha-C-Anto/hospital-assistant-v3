
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router

from app.services.rag_service import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.rag_service = RAGService()

    yield

app = FastAPI(lifespan=lifespan)
 
app.include_router(router)