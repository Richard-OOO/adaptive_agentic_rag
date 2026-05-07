import logging
from typing import Optional, Dict
from langchain_openai import ChatOpenAI

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMManager:
    _instances: Dict[str, ChatOpenAI] = {}
    _default_instance: Optional[ChatOpenAI] = None

    @classmethod
    def get_default_llm(cls) -> ChatOpenAI:
        if cls._default_instance is None:
            settings = get_settings()
            cls._default_instance = cls.get_llm(
                temperature=0.0,
                max_tokens=settings.openai_max_tokens,
                model=settings.openai_model,
                base_url=settings.openai_api_base,
                api_key=settings.openai_api_key,
            )
        return cls._default_instance

    @classmethod
    def get_llm(
        cls,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        streaming: bool = False,
    ) -> ChatOpenAI:
        settings = get_settings()

        resolved_base_url = base_url or settings.openai_api_base
        resolved_model = model or settings.openai_model
        resolved_api_key = api_key or settings.openai_api_key

        cache_key = f"{resolved_base_url}|{resolved_model}|{temperature}|{max_tokens}|{streaming}"

        if cache_key not in cls._instances:
            logger.info(
                f"[LLMManager] 创建新的LLM实例: base_url={resolved_base_url}, "
                f"model={resolved_model}, temp={temperature}, max_tokens={max_tokens}, streaming={streaming}"
            )
            cls._instances[cache_key] = ChatOpenAI(
                base_url=resolved_base_url,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=resolved_api_key,
                streaming=streaming,
            )

        return cls._instances[cache_key]

    @classmethod
    def clear_all(cls):
        cls._instances.clear()
        cls._default_instance = None
        logger.info("[LLMManager] 已清空所有LLM实例")


if __name__ == "__main__":
    llm = LLMManager.get_default_llm()
    print(llm.invoke([{"role": "user", "content": "你好"}]))
