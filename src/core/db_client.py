import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class RedisClientManager:
    _redis_pool = None
    _redis_client = None

    @classmethod
    async def init_connections(cls, redis_url: str = "redis://localhost:6379/0"):
        try:
            from redis.asyncio import Redis, ConnectionPool
            from redis.exceptions import ConnectionError as RedisConnectionError
        except ImportError:
            logger.warning("[DB Client] redis 库未安装，跳过 Redis 初始化")
            return

        if cls._redis_client is not None:
            return
        try:
            cls._redis_pool = ConnectionPool.from_url(redis_url, decode_responses=False)
            cls._redis_client = Redis(connection_pool=cls._redis_pool)
            await cls._redis_client.ping()
            logger.info("[DB Client] Redis 全局连接已建立.")
        except RedisConnectionError as e:
            raise RuntimeError(f"[DB Client] Redis 连接失败: {e}")

    @classmethod
    def get_redis_client(cls):
        if cls._redis_client is None:
            raise RuntimeError("[DB Client] Redis 尚未初始化")
        return cls._redis_client

    @classmethod
    async def close_connections(cls):
        if cls._redis_client:
            await cls._redis_client.close()
            logger.info("[DB Client] Redis 连接已断开.")


class MongoDBClientManager:
    _client: Optional[AsyncIOMotorClient] = None

    @classmethod
    async def init_connections(cls, uri: str):
        if cls._client is not None:
            return
        try:
            cls._client = AsyncIOMotorClient(uri)
            await cls._client.admin.command("ping")
            logger.info("[DB Client] MongoDB 全局连接已建立.")
        except Exception as e:
            raise RuntimeError(f"[DB Client] MongoDB 连接失败: {e}")

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            raise RuntimeError("[DB Client] MongoDB 尚未初始化")
        return cls._client

    @classmethod
    async def close_connections(cls):
        if cls._client:
            cls._client.close()
            logger.info("[DB Client] MongoDB 连接已断开.")
