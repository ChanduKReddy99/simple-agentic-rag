import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.agent.graph import AgenticRAGGraph
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.embeddings import build_embeddings
from app.rag.vectorstore import build_vectorstore
from app.rag.retriever import RetrieverService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, run_name="fastapi_app")

    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings, embeddings)
    retriever = RetrieverService(vectorstore=vectorstore, settings=settings)
    app.state.agent = AgenticRAGGraph(retriever=retriever, settings=settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the FastAPI Agentic RAG server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
