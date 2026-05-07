import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.main import app_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"
    session_id: str = "default_session"


class ChatResponse(BaseModel):
    answer: str
    retrieval_grade: str
    documents: List[Dict[str, Any]]


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if app_state.agent_app is None:
        raise HTTPException(status_code=503, detail="Agent App 尚未初始化，请稍后重试")

    config = {
        "configurable": {
            "thread_id": req.session_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
        }
    }

    try:
        from langgraph.types import Overwrite
    except ImportError:
        class Overwrite:
            def __init__(self, value):
                self.value = value

    inputs = {
        "question": req.question,
        "messages": [("human", req.question)],
        "retrieve_loop_step": Overwrite(0),
        "web_search_loop_step": Overwrite(0),
        "past_search_queries": Overwrite([]),
        "retrieval_grade": "none",
        "matched_domain": ""
    }

    try:
        logger.info(f"[Chat API] 收到问题: {req.question}, Session: {req.session_id}")

        final_state = await app_state.agent_app.ainvoke(inputs, config=config)

        answer = final_state.get("generation", "未生成回答")
        grade = final_state.get("retrieval_grade", "none")

        docs = final_state.get("documents", [])
        formatted_docs = []
        for d in docs:
            formatted_docs.append({
                "content": d.page_content[:500],
                "metadata": d.metadata
            })

        return ChatResponse(
            answer=answer,
            retrieval_grade=grade,
            documents=formatted_docs
        )

    except Exception as e:
        logger.error(f"[Chat API] 调用失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
