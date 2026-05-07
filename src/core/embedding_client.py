
import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RemoteEmbeddingClient:

    _client: httpx.AsyncClient = None

    def __init__(self, api_url: str):
        self.api_url = api_url

    @classmethod
    def _get_http_client(cls) -> httpx.AsyncClient:
        """
        单例模式维护 HTTP Keep-Alive 连接池
        """
        if cls._client is None:
            
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
            
            cls._client = httpx.AsyncClient(timeout=120.0, limits=limits)
            
            logger.info("[EmbeddingClient] 已初始化全局 HTTP 连接池")
        return cls._client

    async def aencode(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        异步调用远程微服务，返回双向量
        """
        if not texts:
            return []

        client = self._get_http_client()
        try:
            
            response = await client.post(
                f"{self.api_url}/v1/bge-m3/encode",
                json={"texts": texts}
            )
            response.raise_for_status()
            return response.json()["data"]
        except httpx.HTTPError as e:
            logger.error(f"[EmbeddingClient] 远程向量化调用失败: {e}")
            raise

    def encode(self, texts: List[str]) -> List[Dict[str, Any]]:
        
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
           
            raise RuntimeError("在异步上下文中请调用 aencode()")
        else:
            return asyncio.run(self.aencode(texts))

    @classmethod
    async def close(cls):
       
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None
            logger.info("[EmbeddingClient] HTTP 连接池已关闭")

