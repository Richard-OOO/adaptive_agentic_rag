import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.core.config import get_settings
from src.core.db_client import MongoDBClientManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query & Management"])

@router.get("/milvus/query")
async def query_milvus(
    query: str = Query(..., description="查询文本"),
    top_k: int = Query(5, description="返回数量"),
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
):
    try:
        from src.retrieval.hybrid import NativeMilvusSearcher
        settings = get_settings()
        searcher = NativeMilvusSearcher(
            collection_name=settings.milvus_collection_name,
            milvus_uri=settings.milvus_uri,
            embedding_api_url=settings.embedding_api_url,
        )
        docs = await searcher.asearch(query, k=top_k, user_id=user_id, session_id=session_id)
        return {
            "status": "ok",
            "total": len(docs),
            "results": [
                {
                    "text": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in docs
            ],
        }
    except Exception as e:
        logger.error(f"[Milvus Query] 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/milvus/count")
async def milvus_count():
    try:
        from src.core.vector_client import MilvusClientManager
        settings = get_settings()
        client = MilvusClientManager.get_client(uri=settings.milvus_uri)
        stats = client.get_collection_stats(settings.milvus_collection_name)
        return {"status": "ok", "collection": settings.milvus_collection_name, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/milvus/collection")
async def delete_milvus_collection():
    try:
        from src.core.vector_client import MilvusClientManager
        settings = get_settings()
        client = MilvusClientManager.get_client(uri=settings.milvus_uri)
        if client.has_collection(settings.milvus_collection_name):
            client.drop_collection(settings.milvus_collection_name)
            MilvusClientManager.close_client()
            return {"status": "ok", "message": f"集合 {settings.milvus_collection_name} 已删除"}
        return {"status": "ok", "message": "集合不存在"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mongo/get/{node_id}")
async def get_mongo_payload(
    node_id: str,
    user_id: Optional[str] = Query(None),
):
    try:
        settings = get_settings()
        mongo_client = MongoDBClientManager.get_client()
        db = mongo_client[settings.mongodb_db_name]
        collection = db[settings.mongodb_collection_name]
        doc_id = f"rag_payload:{user_id or 'default'}:{node_id}"
        doc = await collection.find_one({"_id": doc_id})
        if doc is None:
            return {"status": "not_found", "key": doc_id}
        doc.pop("_id", None)
        return {"status": "ok", "key": doc_id, "data": doc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mongo/list")
async def list_mongo_payloads(
    user_id: Optional[str] = Query(None),
    limit: int = Query(100),
):
    try:
        settings = get_settings()
        mongo_client = MongoDBClientManager.get_client()
        db = mongo_client[settings.mongodb_db_name]
        collection = db[settings.mongodb_collection_name]
        query = {}
        if user_id:
            query["_id"] = {"$regex": f"^rag_payload:{user_id}:"}
        cursor = collection.find(query, {"_id": 1}).limit(limit)
        docs = await cursor.to_list(length=limit)
        keys = [d["_id"] for d in docs]
        return {"status": "ok", "total": len(keys), "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/mongo/key/{key:path}")
async def delete_mongo_key(key: str):
    try:
        settings = get_settings()
        mongo_client = MongoDBClientManager.get_client()
        db = mongo_client[settings.mongodb_db_name]
        collection = db[settings.mongodb_collection_name]
        result = await collection.delete_one({"_id": key})
        return {"status": "ok", "key": key, "deleted": result.deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
