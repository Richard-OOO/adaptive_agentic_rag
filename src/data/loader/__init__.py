from .base import BaseLoader
from .docx_loader import DocxLoader
from .factory import (
    get_modality,
    tag_modality,
    tag_modality_batch,
    inject_user_session,
    inject_user_session_batch,
    MODALITY_MAP,
    EXTENSION_TO_MODALITY,
)

__all__ = [
    "BaseLoader",
    "DocxLoader",
    "get_modality",
    "tag_modality",
    "tag_modality_batch",
    "inject_user_session",
    "inject_user_session_batch",
    "MODALITY_MAP",
    "EXTENSION_TO_MODALITY",
]
