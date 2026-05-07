import logging
from typing import Optional
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class MilvusClientManager:
    _client: Optional[MilvusClient] = None

    @classmethod
    def get_client(cls, uri: str) -> MilvusClient:
        if cls._client is not None:
            return cls._client

        try:
            cls._client = MilvusClient(uri=uri)
            logger.info(f"[VectorClient] MilvusClient 已连接 uri={uri}")
            return cls._client
        except Exception as exc:
            logger.error(f"[VectorClient] 连接失败: {exc}")
            raise

    @classmethod
    def close_client(cls) -> None:
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            logger.info("[VectorClient] 连接已关闭")


def get_milvus_client(uri: str) -> MilvusClient:
    return MilvusClientManager.get_client(uri)
