import logging
import httpx
from typing import List
import concurrent.futures
logger = logging.getLogger(__name__)

class RerankClient:
    
    _client: httpx.AsyncClient = None

    def __init__(self, api_base: str, model_name: str, api_key: str = "EMPTY"):
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key


    @classmethod
    def _get_http_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
            cls._client = httpx.AsyncClient(timeout=120.0, limits=limits)
            logger.info("[RerankClient] 初始化全局HTTP 连接池")
        return cls._client

    async def arerank(self, query:str, texts: List[str]) -> List[float]:
        print(query, texts)
        if not texts:
            return []

        client = self._get_http_client()

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key and self.api_key.strip() and self.api_key.strip() != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key.strip()}"
        
        # payload = {
        #     "model": self.model_name,

        #     "query": query,
        #     "documents": texts
            
        # }
        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": texts
            }

        }

        try:
            response = await client.post(
                f"{self.api_base}", 
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            data = data.get("output", [])
            print(data)
            scores = [0.0] * len(texts)
            
            for res in data.get("results", []):

                idx = res.get("index")
                if idx is not None and 0 <= idx < len(scores):

                    scores[idx] = res.get("relevance_score", res.get("score", 0.0))
            
            return scores
            
        except httpx.HTTPError as e:
            logger.error(f"API 调用失败: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"响应详情: {e.response.text}")

            return [0.0] * len(texts)

if __name__ == "__main__":
    import asyncio
    from src.core.config import Settings
    
    settings = Settings()
    
    reranker_client = RerankClient(
                api_base=settings.reranker_api_base,
                model_name=settings.reranker_model_name,
                api_key=settings.reranker_api_key,
            )
    scores=asyncio.run(reranker_client.arerank("心理学作为独立学科诞生于哪一年？", ["心理学作为独立学科诞生于1920年", "我今天去吃饭"]))
    print(scores)    