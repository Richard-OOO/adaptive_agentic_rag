import re
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ProseStrategy:
    """散文分块策略
    
    使用 RecursiveCharacterTextSplitter.from_tiktoken_encoder 按 Token 数切分。
    支持 Parent-Child 多粒度分块，并执行分块后的微观清洗。
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 250,
        child_chunk_overlap: int = 30,
    ):
        """初始化散文分块策略
        
        Args:
            parent_chunk_size: 父节点块大小（tokens），默认 1000
            parent_chunk_overlap: 父节点重叠大小，默认 100
            child_chunk_size: 子节点块大小（tokens），默认 250
            child_chunk_overlap: 子节点重叠大小，默认 30
        """
        self.parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "。", "；", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "。", "；", ""]
        )
    
    def _post_clean(self, chunk: Document) -> Optional[Document]:
        """分块后的微观打磨
        
        清洗规则：
        1. strip() 去除首尾空白
        2. 长度 < 15 丢弃
        3. 有意义字符（中文/字母/数字）占比低于 0.5 丢弃
        4. 剔除开头孤立的闭合括号
        
        Args:
            chunk: LangChain Document 对象
        
        Returns:
            清洗后的 Document 或 None（表示应丢弃）
        """
        if not chunk or not chunk.page_content:
            return None
        
        content = chunk.page_content.strip()
        
        if len(content) < 15:
            return None
        
        meaningful_chars = re.findall(r'[\u4e00-\u9fff\w]', content)
        content_len = len(content)
        if len(meaningful_chars) / content_len < 0.2:
            return None
        
        content = re.sub(r'^[\)\]\}]+', '', content)
        content = content.strip()
        
        if len(content) < 15:
            return None
        
        chunk.page_content = content
        return chunk
    
    def process(self, doc: Document) -> List[Document]:
        """处理散文文档，生成多粒度层级映射
        
        流程：
        1. 切分父节点（大块）
        2. 对每个父节点：
           a. 先切分子节点（小块）—— 必须在父节点打标签之前！
           b. 为父节点生成 UUID，打上 level="large"
           c. 为子节点打上 level="small" 和 parent_id
        3. 对所有块执行 post_clean 过滤
        
        Args:
            doc: LangChain Document 对象
        
        Returns:
            分块后的 Document 列表（包含父节点和子节点）
        """
        if not doc or not doc.page_content:
            return []
        
        parent_docs = self.parent_splitter.split_documents([doc])
        
        result: List[Document] = []
        
        for parent_doc in parent_docs:
            parent_id = str(uuid.uuid4())
            
            child_docs = self.child_splitter.split_documents([parent_doc])
            
            parent_doc.metadata["node_id"] = parent_id
            parent_doc.metadata["level"] = "large"
            
            cleaned_parent = self._post_clean(parent_doc)
            if cleaned_parent:
                result.append(cleaned_parent)
            
            for child_doc in child_docs:
                child_doc.metadata["parent_id"] = parent_id
                child_doc.metadata["level"] = "small"
                
                cleaned_child = self._post_clean(child_doc)
                if cleaned_child:
                    result.append(cleaned_child)
        
        return result
