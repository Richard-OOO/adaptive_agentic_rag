import json
import logging
import asyncio
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from pymongo import UpdateOne

from src.core.config import get_settings
from src.core.embedding_client import RemoteEmbeddingClient
from src.core.vector_client import MilvusClientManager
from src.core.db_client import MongoDBClientManager
from pymilvus import DataType
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataIndexer:
    def __init__(
        self,
        mongodb_db_name: str,
        mongodb_collection_name: str,
        milvus_uri: str,
        collection_name: str,
        embedding_api_url: str,
        vector_dim: int = 1024,
    ):
        self.collection_name = collection_name
        self.client = MilvusClientManager.get_client(uri=milvus_uri)
        self.embedding_client = RemoteEmbeddingClient(api_url=embedding_api_url)
        self.executor = ThreadPoolExecutor(max_workers=4)

        if not self.client.has_collection(self.collection_name):
            self._setup_collection(vector_dim)
        self.client.load_collection(self.collection_name)

        mongo_client = MongoDBClientManager.get_client()
        self.mongo_db_name = mongodb_db_name
        self.mongo_collection_name = mongodb_collection_name
        self.mongo_collection = mongo_client[mongodb_db_name][mongodb_collection_name]

    @classmethod
    def from_settings(cls, vector_dim: int = 1024) -> "DataIndexer":
        settings = get_settings()
        return cls(
            mongodb_db_name=settings.mongodb_db_name,
            mongodb_collection_name=settings.mongodb_collection_name,
            milvus_uri=settings.milvus_uri,
            collection_name=settings.milvus_collection_name,
            embedding_api_url=settings.embedding_api_url,
            vector_dim=vector_dim,
        )

    def _setup_collection(self, vector_dim):
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)

    def _validate_and_split_sync(self, docs: List[Document]) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        search_docs = []
        payload_kv_pairs = []
        for doc in docs:
            level = doc.metadata.get("level")
            if level == "large":
                node_id = doc.metadata.get("node_id")
                if node_id:
                    payload_kv_pairs.append((node_id, doc))
            else:
                search_docs.append(doc)
        return search_docs, payload_kv_pairs

    async def aindex(self, docs: List[Document], batch_size: int = 20, user_id: str = None, session_id: str = None) -> Dict[str, int]:
        if not docs:
            return {"vector_inserts": 0, "kv_inserts": 0}

        loop = asyncio.get_running_loop()

        search_docs, payload_kv_pairs = await loop.run_in_executor(
            self.executor, self._validate_and_split_sync, docs
        )

        if payload_kv_pairs:
            mongo_ops = []
            for k, doc in payload_kv_pairs:
                doc_id = f"rag_payload:{user_id or 'default'}:{k}"
                doc_data = doc.dict()
                mongo_ops.append(
                    UpdateOne(
                        {"_id": doc_id},
                        {"$set": doc_data},
                        upsert=True
                    )
                )
            if mongo_ops:
                await self.mongo_collection.bulk_write(mongo_ops)

        vector_inserted = 0
        if search_docs:
            for i in range(0, len(search_docs), batch_size):
                batch = search_docs[i:i + batch_size]
                texts = [doc.page_content for doc in batch]

                embeddings = await self.embedding_client.aencode(texts)

                insert_data = []
                for doc, emb in zip(batch, embeddings):
                    row = {"dense_vector": emb["dense"], "sparse_vector": emb["sparse"], "text": doc.page_content}
                    meta = dict(doc.metadata)
                    if "knowledge_domains" in meta and isinstance(meta["knowledge_domains"], list):
                        meta["knowledge_domains"] = ",".join(meta["knowledge_domains"])
                    row.update(meta)
                    if user_id:
                        row["user_id"] = user_id
                    if session_id:
                        row["session_id"] = session_id
                    insert_data.append(row)

                await loop.run_in_executor(
                    self.executor,
                    self.client.insert,
                    self.collection_name,
                    insert_data
                )
                vector_inserted += len(batch)

        return {"vector_inserts": vector_inserted, "kv_inserts": len(payload_kv_pairs)}
