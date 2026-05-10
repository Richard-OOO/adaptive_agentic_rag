import logging
import asyncio
from typing import List, Optional

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, WeightedRanker

from src.core.vector_client import MilvusClientManager
from src.core.embedding_client import RemoteEmbeddingClient

logger = logging.getLogger(__name__)


class NativeMilvusSearcher:
    def __init__(
        self,
        collection_name: str,
        milvus_uri: str,
        embedding_api_url: str,
        embedding_model_name: str,
        embedding_api_key: str,
    ):
        self.collection_name = collection_name
        self.milvus_uri = milvus_uri
        self.client = MilvusClientManager.get_client(uri=milvus_uri)
        self.embedding_client = RemoteEmbeddingClient(api_url=embedding_api_url, model_name=embedding_model_name, api_key=embedding_api_key)

        try:
            self.client.load_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"加载集合失败 (可能还未创建): {e}")

    def _build_expr(self, user_id: str | None, session_id: str | None, knowledge_domain: str | None = None) -> str | None:
        parts = []
        if user_id:
            parts.append(f'user_id == "{user_id}"')
        if session_id:
            parts.append(f'session_id == "{session_id}"')
        if knowledge_domain:
            parts.append(f'knowledge_domains like "%{knowledge_domain}%"')
        return " and ".join(parts) if parts else None

    async def asearch(self, query: str, k: int = 5, user_id: str | None = None, session_id: str | None = None, knowledge_domain: str | None = None) -> List[Document]:
        expr = self._build_expr(user_id, session_id, knowledge_domain)

        query_embs = await self.embedding_client.aencode([query])
        if not query_embs:
            return []

        query_vecs = query_embs[0]

        def _do_hybrid_search():
            req_dense = AnnSearchRequest(
                data=[query_vecs["embedding"]],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=k * 2,
                expr=expr
            )

            req_sparse = AnnSearchRequest(
                data=[query_vecs["sparse_embedding"]],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=k * 2,
                expr=expr
            )

            return self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[req_dense, req_sparse],
                ranker=WeightedRanker(0.7, 0.3),
                limit=k,
                output_fields=["*"]
            )

        results = await asyncio.to_thread(_do_hybrid_search)

        docs = []
        if not results or not results[0]:
            return docs

        for hit in results[0]:
            entity = hit.get("entity", {})
            text = entity.pop("text", "")
            entity["hybrid_score"] = hit.get("distance", 0.0)
            docs.append(Document(page_content=text, metadata=entity))

        return docs
