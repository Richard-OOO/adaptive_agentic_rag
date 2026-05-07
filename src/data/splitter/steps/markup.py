
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class MarkupStrategy:
    """标记语言分块策略
    
    针对 Markdown 等标记语言，优先按标题层级切分。
    不需要多粒度切分，也不需要 post_clean。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        strip_headers: bool = False,
    ):
        """初始化标记语言分块策略
        
        Args:
            chunk_size: 块大小（字符数），默认 1000
            chunk_overlap: 重叠大小，默认 100
            strip_headers: 是否从内容中移除标题文本，默认 False
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_headers = strip_headers
        
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ],
            strip_headers=strip_headers,
        )
        
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ""]
        )
    
    def _is_markdown(self, doc: Document) -> bool:
        """判断文档是否为 Markdown 格式
        
        根据 metadata["source"] 的后缀判断。
        
        Args:
            doc: LangChain Document 对象
        
        Returns:
            是否为 Markdown 文档
        """
        source = doc.metadata.get("source", "")
        return source.lower().endswith((".md", ".markdown"))
    
    def process(self, doc: Document) -> List[Document]:
        """处理标记语言文档
        
        流程：
        1. 如果是 Markdown，优先按标题层级切分
        2. 对切分后的块进行二次切分（如果超过 chunk_size）
        3. 为每个块打上 level="markup_chunk" 和 node_id
        
        Args:
            doc: LangChain Document 对象
        
        Returns:
            分块后的 Document 列表
        """
        if not doc or not doc.page_content:
            return []
        
        result: List[Document] = []
        
        if self._is_markdown(doc):
            try:
                chunks = self._markdown_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata["node_id"] = str(uuid.uuid4())
                    chunk.metadata["level"] = "markup_chunk"
                    result.append(chunk)
            except Exception:
                chunks = self._fallback_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["node_id"] = str(uuid.uuid4())
                    chunk.metadata["level"] = "markup_chunk"
                    result.append(chunk)
        else:
            chunks = self._fallback_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["node_id"] = str(uuid.uuid4())
                chunk.metadata["level"] = "markup_chunk"
                result.append(chunk)
        
        return result
