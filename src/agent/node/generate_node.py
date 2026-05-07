import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig  
from src.core.llm_manager import LLMManager         
from langchain_core.messages import AIMessage, trim_messages
from src.agent.state import GraphState
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RETRIEVE_GENERATE_SYSTEM_PROMPT = """你是一个智能问答助手。
请优先使用提供的文档上下文来回答用户问题。
如果上下文中提供的信息不足以完整回答问题，你可以结合你自身的知识进行补充，但请在回答中明确指出哪些部分来源于上下文，哪些部分是你补充的常识。
"""
TALK_GENERATE_SYSTEM_PROMPT = """你是一个专业的回答用户问题的助手。
认真回答用户的问题，不要编造信息。
"""
def _format_documents_context(docs: List[Document], max_chars_per_doc: int = 1200) -> str:
   
    parts = []
    for i, d in enumerate(docs, start=1):
        title = (d.metadata or {}).get("title") or (d.metadata or {}).get("source") or f"doc_{i}"
        content = (d.page_content or "").strip().replace("\n", " ")[:max_chars_per_doc]
        parts.append(f"--- Document {i}: {title}\n{content}\n")
    return "\n".join(parts)



async def generate_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    question = state.get("question")
    if not question:
        raise ValueError("[Generate Node] state 缺失 question")
    messages = state.get("messages", [])
    documents = state.get("documents", []) or []
    need_retrieval=state.get("need_retrieval", True)
    docs_ctx = _format_documents_context(documents)
    configurable = config.get("configurable", {})
    llm = LLMManager.get_llm()
    if not need_retrieval:
        prompt=ChatPromptTemplate.from_messages([
        ("system", TALK_GENERATE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "问题: {question}\n\n请直接回答问题。")
    ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", RETRIEVE_GENERATE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "问题: {question}\n\n上下文:\n{documents}\n\n请基于以上上下文直接回答问题。")
        ])


    trimmed_messages = trim_messages(
        messages,
        max_tokens=50,
        strategy="last",
        token_counter=len,
        include_system=True,
    )

    chain = prompt | llm

    try:
        response_text = await chain.ainvoke({"messages": trimmed_messages, "question": question, "documents": docs_ctx})
        
        # Check fallback scenario
        potential_hallucination = state.get("retrieval_grade") == "no"
        
        return {
            "generation": response_text.content if hasattr(response_text, "content") else str(response_text),
            "potential_hallucination": potential_hallucination,
            "messages": [AIMessage(content=response_text.content)]
        }
    except Exception as e:
        
        logger.error(f"[Generate Node] 生成失败: {e}")
        return {"generation": "根据已知信息无法回答", "potential_hallucination": True}
