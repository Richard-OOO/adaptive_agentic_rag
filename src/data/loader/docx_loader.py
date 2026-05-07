from pathlib import Path
from typing import Union, List
import re

try:
    from .base import BaseLoader
except ImportError:
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent))
    from base import BaseLoader


class DocxLoader(BaseLoader):
    def __init__(self, remove_header_footer: bool = False):
        self.remove_header_footer = remove_header_footer
        
    def load(self, file_path: Union[str, Path]) -> List:

        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if path.suffix.lower() not in ['.docx', '.doc']:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        docs = self._extract_content(path)
        
        for doc in docs:
            doc.page_content = self._clean_text(doc.page_content)
            doc.metadata.update({
                'format': 'docx',
                'filename': path.name,
                'file_size': path.stat().st_size,
            })
        
        return docs
    
    def _extract_content(self, path: Path) -> List:
        """提取Word文档内容，返回LangChain Document列表"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            return docs
            
        except Exception as e:
            raise ImportError(f"加载Word文档时出错: {e}")
    
    def _clean_text(self, text: str) -> str:
        """轻度清洗 - 智能处理换行符"""
        if not text:
            return ""
        text = text.replace('\r\n', '\n')
        text = self._remove_tabs(text)
        text = self._clean_newlines(text)
        
        if self.remove_header_footer:
            text = self._remove_header_footer(text)
        
        return text.strip()
    
    def _clean_newlines(self, text: str) -> str:
        """智能处理换行符
        
        规则：
        1. 统一3+换行为双换行（保留段落结构）
        2. 保留所有双换行（最高级结构）
        3. 对单换行进行智能判断
        """
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        paragraphs = re.split(r'\n\n', text)
        result = []
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            if i > 0:
                result.append('\n\n')
            
            para_result = self._process_single_newlines(para)
            result.append(para_result)
        
        return ''.join(result)
    
    def _process_single_newlines(self, text: str) -> str:
        """处理段落内的单换行
        
        Args:
            text: 段落文本
        
        Returns:
            处理后的文本
        """
        lines = text.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            if i > 0:
                prev_line = lines[i - 1]
                
                if prev_line.strip():
                    if self._should_keep_newline(prev_line, line):
                        result.append('\n')
                    else:
                        result.append('')
            
            result.append(line)
        
        return ''.join(result)
    
    def _should_keep_newline(self, prev_line: str, next_line: str) -> bool:
        """判断是否应该保留换行
        
        Args:
            prev_line: 前一行
            next_line: 后一行
        
        Returns:
            True=保留换行，False=缝合
        """
        end_punctuations = {'。', '！', '？', '.', '!', '?', '；', ';'}
        
        if prev_line and prev_line[-1] in end_punctuations:
            return True
        
        list_patterns = [
            r'^\d+\.',
            r'^[-*•]',
            r'^\(\d+\)',
            r'^[一二三四五六七八九十]'
        ]
        
        for pattern in list_patterns:
            if re.match(pattern, next_line.strip()):
                return True
        
        return False
    
    def _remove_tabs(self, text: str) -> str:
        """去除制表符"""
        text = text.replace('\t', ' ')
        text = re.sub(r' +', ' ', text)
        return text
    
    def _remove_header_footer(self, text: str) -> str:
        """去除页眉页脚
        
        规则：
        1. 匹配常见页脚格式：第X页、Page X、X/Y
        2. 匹配行首/行尾孤立数字
        3. 匹配末尾孤立的数字（如 " 2"）
        """
        patterns = [
            r'^第\s*\d+\s*页\s*$',
            r'^Page\s*\d+\s*$',
            r'^\d+\s*/\s*\d+\s*$',
            r'^-+\s*\d+\s*-+$',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\n\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\n\s*\d+\s*(?=\n)', '\n', text)
        
        text = re.sub(r'\n\s{1,3}\d{1,3}\s*$', '', text)
        
        return text


if __name__ == "__main__":
    loader = DocxLoader()
    docs = loader.load("C:\\Users\\28233\\Desktop\\11111.docx")
    
    for i, doc in enumerate(docs):
        print(f"文档 {i+1}:")
        print(f"元数据: {doc.metadata}")
        print(f"内容长度: {len(doc.page_content)} 字符")
        print(f"内容预览:\n{doc.page_content[:200]}...")
        print("-" * 50)
