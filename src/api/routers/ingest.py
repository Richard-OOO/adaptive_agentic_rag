import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from src.data.loader.docx_loader import DocxLoader
from src.data.loader.factory import tag_modality_batch, inject_user_session_batch, inject_knowledge_domains_batch
from src.api.main import app_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    file_path: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    slice_start: Optional[int] = None
    slice_end: Optional[int] = None
    domains: Optional[List[str]] = None


class IngestResponse(BaseModel):
    status: str
    file_path: str
    total_docs_loaded: int
    total_chunks: int
    vector_inserts: int
    kv_inserts: int
    

@router.post("/docx", response_model=IngestResponse)
async def ingest_docx(req: IngestRequest):
    if app_state.pipeline is None or app_state.indexer is None:
        raise HTTPException(status_code=503, detail="Pipeline 或 Indexer 尚未初始化，请稍后重试")

    try:
        loader = DocxLoader()
        docs = loader.load(req.file_path)

        if req.slice_start is not None or req.slice_end is not None:
            start = req.slice_start or 0
            end = req.slice_end or len(docs)
            docs = docs[start:end]

        total_loaded = len(docs)

        docs = tag_modality_batch(docs, req.file_path)
        docs = inject_user_session_batch(docs, user_id=req.user_id, session_id=req.session_id)
        docs = inject_knowledge_domains_batch(docs, domains=req.domains)



        chunks = app_state.pipeline.process_batch(docs)
        total_chunks = len(chunks)

        result = await app_state.indexer.aindex(chunks, user_id=req.user_id, session_id=req.session_id)

        return IngestResponse(
            status="ok",
            file_path=req.file_path,
            total_docs_loaded=total_loaded,
            total_chunks=total_chunks,
            vector_inserts=result["vector_inserts"],
            kv_inserts=result["kv_inserts"],
        )
    except Exception as e:
        logger.error(f"[Ingest] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
