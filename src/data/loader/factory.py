from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document


# Temporary modality mapping: code file extensions are treated as 'prose'
MODALITY_MAP: Dict[str, set] = {
    "prose": {".docx", ".doc", ".pdf", ".txt", ".rtf", ".odt"},
    # "code": {        # code extensions currently mapped to prose for plain-text mode
    #           ".py", ".js", ".ts", ".jsx", ".tsx",
    #           ".java", ".kt", ".kts", ".scala",
    #           ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    #           ".cs", ".go", ".rs", ".rb", ".php",
    #           ".swift", ".m", ".mm",
    #           ".sh", ".bash", ".zsh", ".ps1",
    #           ".sql", ".json", ".yaml", ".yml", ".toml",},
    "markup": {".md", ".markdown", ".html", ".htm", ".xml", ".rst"},
}

EXTENSION_TO_MODALITY: Dict[str, str] = {}
for modality, extensions in MODALITY_MAP.items():
    for ext in extensions:
        EXTENSION_TO_MODALITY[ext.lower()] = modality


def get_modality(file_path: str) -> str:
    
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_MODALITY.get(ext, "prose")


def tag_modality(doc: Document, file_path: Optional[str] = None) -> Document:

    source = file_path or doc.metadata.get("source", "")
    modality = get_modality(source) if source else "prose"
    doc.metadata["modality"] = modality
    return doc

def tag_modality_batch(docs: List[Document], file_path: Optional[str] = None) -> List[Document]:
    for doc in docs:
        tag_modality(doc, file_path)
    return docs


def inject_user_session(doc: Document, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Document:
    """在 loader 层注入 user_id / session_id 到 Document.metadata 中。"""
    if not isinstance(doc.metadata, dict):
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    if user_id:
        doc.metadata["user_id"] = user_id
    if session_id:
        doc.metadata["session_id"] = session_id
    return doc


def inject_user_session_batch(docs: List[Document], user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Document]:
    for doc in docs:
        inject_user_session(doc, user_id=user_id, session_id=session_id)
    return docs


def inject_knowledge_domains(doc: Document, domains: Optional[List[str]] = None) -> Document:
    if not isinstance(doc.metadata, dict):
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    if domains:
        doc.metadata["knowledge_domains"] = domains
    return doc


def inject_knowledge_domains_batch(docs: List[Document], domains: Optional[List[str]] = None) -> List[Document]:
    for doc in docs:
        inject_knowledge_domains(doc, domains=domains)
    return docs