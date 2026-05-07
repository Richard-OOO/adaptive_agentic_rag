import logging
import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from src.agent.state import GraphState
from src.retrieval.hybrid import NativeMilvusSearcher
from src.core.db_client import MongoDBClientManager
from src.core.reranker_client import RerankClient
from src.core.config import get_settings

logger = logging.getLogger(__name__)


async def retrieve_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:

    queries: List[str] = state.get("search_queries", []) or []
    requires_extended = state.get("requires_extended_context", False)
    original_question = state.get("question", "")
    matched_domain = state.get("matched_domain", "")

    if not queries:
        return {"documents": []}

    configurable = config.get("configurable", {})
    settings = get_settings()

    user_id = configurable.get("user_id", "")
    session_id = configurable.get("session_id", "")
    top_k = configurable.get("search_top_k", 5)
    rerank_top_k = configurable.get("rerank_top_k", 3)

    searcher = NativeMilvusSearcher(
        collection_name=settings.milvus_collection_name,
        milvus_uri=settings.milvus_uri,
        embedding_api_url=settings.embedding_api_url,
    )
    agg: List[Document] = []

    for q in queries:
        try:
            docs = await searcher.asearch(
                q, k=top_k, user_id=user_id, session_id=session_id,
                knowledge_domain=matched_domain if matched_domain else None
            )
            agg.extend(docs)
        except Exception as e:
            logger.warning(f"[Retrieve Node] 检索查询失败 for '{q}': {e}")
    if not agg:
        return {"documents":[], "retrieve_loop_step": 10}
    seen = set()
    unique_docs: List[Document] = []
    for d in agg:
        nid = d.metadata.get("node_id") if d.metadata else None
        key = (nid, d.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(d)

    if requires_extended and unique_docs:
        logger.info(f"[Retrieve Node] 触发宏观上下文扩展，准备从 数据库 提取 {len(unique_docs)} 个大块。")
        try:
            mongo_client=MongoDBClientManager.get_client()
            
            db = mongo_client[settings.mongodb_db_name]
            mongo_collection = db[settings.mongodb_collection_name]
            
            keys_to_fetch = []
           
            doc_idx_map = {}

            for i, doc in enumerate(unique_docs):
                meta = doc.metadata or {}
                target_id = meta.get("parent_id") or meta.get("node_id")
                if target_id:
                    mongo_id = f"rag_payload:{user_id or 'default'}:{target_id}"
                    keys_to_fetch.append(mongo_id)
                    doc_idx_map[mongo_id] = i
            if keys_to_fetch:

                cursor = mongo_collection.find({"_id": {"$in": keys_to_fetch}})
                results = await cursor.to_list(length=len(keys_to_fetch))
                
                
                for mongo_doc in results:
                    _id = mongo_doc["_id"]
                    idx = doc_idx_map.get(_id)
                    
                    if idx is not None:
                        
                        unique_docs[idx].page_content = mongo_doc.get("page_content", unique_docs[idx].page_content)
                        unique_docs[idx].metadata["is_extended_chunk"] = True


        except Exception as e:
            logger.warning(f"[Retrieve Node] 数据库大块提取失败，降级使用 Milvus 小块: {e}")
   
    return {
        "retrieve_loop_step": 1,
        "documents": unique_docs
    }
