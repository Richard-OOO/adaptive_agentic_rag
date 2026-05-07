import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.db_client import MongoDBClientManager

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)


_configure_logging()

settings = get_settings()


class AppState:
    def __init__(self):
        self.pipeline = None
        self.indexer = None
        self.agent_app = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Server] ========== 应用启动 ==========")

    mongo_ok = False
    try:
        await MongoDBClientManager.init_connections(uri=settings.mongodb_uri)
        logger.info("[Server] MongoDB 连接池已建立")
        mongo_ok = True
    except Exception as e:
        logger.warning(f"[Server] MongoDB 连接失败，/ingest /chat 端点不可用: {e}")

    if mongo_ok:
        try:
            from src.data.splitter.pipeline import SplitPipeline
            from src.data.indexer import DataIndexer
            app_state.pipeline = SplitPipeline()
            app_state.indexer = DataIndexer.from_settings()
            logger.info("[Server] Pipeline + Indexer 初始化完成")
        except Exception as e:
            logger.warning(f"[Server] Pipeline/Indexer 初始化失败: {e}")

        try:
            from src.agent.graph import build_compiled_app
            app_state.agent_app = await build_compiled_app()
            if app_state.agent_app is not None:
                logger.info("[Server] Agent Graph App 初始化完成")
            else:
                logger.warning("[Server] Agent Graph App 初始化失败，/chat 端点不可用")
        except Exception as e:
            logger.warning(f"[Server] Agent Graph App 初始化失败: {e}")

    logger.info("[Server] ========== 应用就绪 ==========")

    yield

    logger.info("[Server] ========== 应用关闭 ==========")

    try:
        from src.core.vector_client import MilvusClientManager
        MilvusClientManager.close_client()
    except Exception:
        pass

    try:
        from src.core.embedding_client import RemoteEmbeddingClient
        await RemoteEmbeddingClient.close()
    except Exception:
        pass

    try:
        from src.core.reranker_client import RerankClient
        if RerankClient._client is not None:
            await RerankClient._client.aclose()
            RerankClient._client = None
    except Exception:
        pass

    try:
        await MongoDBClientManager.close_connections()
    except Exception:
        pass

    logger.info("[Server] ========== 应用已停止 ==========")


app = FastAPI(
    title="Adaptive Agentic RAG API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    if elapsed_ms > 3000:
        logger.warning(f"[SlowRequest] {request.method} {request.url.path} took {elapsed_ms:.0f}ms")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[UnhandledError] {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


from src.api.routers.ingest import router as ingest_router
from src.api.routers.query import router as query_router
from src.api.routers.chat import router as chat_router

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(chat_router)


@app.get("/health", tags=["System"])
async def health(request: Request):
    deps = {
        "mongodb": "unknown",
        "milvus": "unknown",
    }

    try:
        mongo_client = MongoDBClientManager.get_client()
        await mongo_client.admin.command("ping")
        deps["mongodb"] = "ok"
    except Exception as e:
        deps["mongodb"] = f"error: {e}"

    try:
        from src.core.vector_client import MilvusClientManager
        client = MilvusClientManager.get_client(uri=settings.milvus_uri)
        if client.has_collection(settings.milvus_collection_name):
            deps["milvus"] = "ok"
        else:
            deps["milvus"] = "collection_not_found"
    except Exception as e:
        deps["milvus"] = f"error: {e}"

    all_ok = all(v == "ok" for v in deps.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "version": "2.1.0",
        "dependencies": deps,
    }
