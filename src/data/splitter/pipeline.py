import warnings
from typing import Callable, Dict, List

from langchain_core.documents import Document

from .steps.prose import ProseStrategy
from .steps.markup import MarkupStrategy


class SplitPipeline:
    """分割管道
    
    极简路由器，纯粹的三项插座。
    根据 doc.metadata["modality"] 路由到不同的分块策略：
    - prose: 散文自然语言流，支持 Parent-Child 多粒度分块
    - markup: 标记语言流（MD/HTML），按标题层级切分
    - code: 结构化代码流，语义感知切分
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 250,
        child_chunk_overlap: int = 30,
        markup_chunk_size: int = 1000,
        markup_chunk_overlap: int = 100,
        
    ):
        """初始化管道
        
        Args:
            parent_chunk_size: 父节点块大小（tokens），默认 1000
            parent_chunk_overlap: 父节点重叠大小，默认 100
            child_chunk_size: 子节点块大小（tokens），默认 250
            child_chunk_overlap: 子节点重叠大小，默认 30
            markup_chunk_size: 标记语言块大小，默认 1000
            markup_chunk_overlap: 标记语言重叠大小，默认 100
            code_chunk_size: 代码块大小，默认 500
            code_chunk_overlap: 代码重叠大小，默认 50
        """
        self._strategies: Dict[str, Callable[[Document], List[Document]]] = {}
        self._prose_strategy = ProseStrategy(
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )
        
        self._markup_strategy = MarkupStrategy(
            chunk_size=markup_chunk_size,
            chunk_overlap=markup_chunk_overlap,
        )
        
        self._register_defaults()
    def _register_defaults(self) -> None:

        self.register_strategy("prose", self._prose_strategy.process)
        self.register_strategy("markup", self._markup_strategy.process)
        # Note: 'code' strategy temporarily not registered by default.
        # Use `register_strategy("code", ...)` to re-enable when needed.

    def register_strategy(self, modality: str, processor: Callable[[Document], List[Document]]) -> None:
       
        if not callable(processor):
            raise ValueError(f"[Pipeline Error] 为 '{modality}' 注册的 processor 必须是可调用对象(Callable)。")
        self._strategies[modality] = processor

    def process(self, doc: Document) -> List[Document]:
        """门面路由入口：根据 modality 动态分发任务"""
        if not doc or not doc.page_content:
            return []
        
        modality = doc.metadata.get("modality", "prose")
        
        
        processor = self._strategies.get(modality)
        
        if not processor:
            warnings.warn(
                f"[Pipeline Warn] 拦截到未知模态 '{modality}'，强制降级触发 'prose' 兜底策略。"
                f"当前已注册模态: {list(self._strategies.keys())}",
                UserWarning,
                stacklevel=2,
            )
            processor = self._strategies.get("prose")
            
            
            if not processor:
                raise RuntimeError("[Pipeline Error] 致命异常：兜底的 'prose' 策略未被注册！")
                
        return processor(doc)
    
    def process_batch(self, docs: List[Document]) -> List[Document]:
        """批量处理入口"""
        result: List[Document] = []
        for doc in docs:
            result.extend(self.process(doc))
        return result