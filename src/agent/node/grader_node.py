import logging
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from src.agent.state import GraphState
from src.core.config import get_settings
from src.core.reranker_client import RerankClient
from src.core.llm_manager import LLMManager
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

THRESHOLD_YES = 0.05

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

GRADER_SYSTEM_PROMPT = """你是一个严格的文档相关性评估员。
请判断以下【检索文档】中，是否包含了回答【用户问题】所必须的客观事实依据。

【核心原则 - 极度重要】：
用户问题中可能包含特定的语气、格式、字数或角色扮演要求（例如“用鲁迅的口吻”、“写一段 Python 代码”、“总结为表格”等）。
作为检索评估员，你必须**完全忽略**这些形式与身份要求！你只负责评估文档中是否包含了目标概念（如“社会惰化”）的**客观知识/事实定义**。只要知识点存在，即刻予以通过。
请仅输出一个 JSON 对象，格式为 {{"binary_score": "yes"}} 或 {{"binary_score": "no"}}。"""
async def grader_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    docs = state.get("documents", [])
    queries: List[str] = state.get("search_queries", []) or []
    settings = get_settings()
    question = state.get("question", "")
    configurable = config.get("configurable", {})        
    rerank_top_k = configurable.get("rerank_top_k", 3)
    if not docs:
        return {
            "retrieval_grade": "no",
            "documents": []
        }
    try:
        texts_to_rerank = [doc.page_content for doc in docs]
        query_to_rerank = question if question else queries[0]

        reranker_client = RerankClient(
                api_base=settings.reranker_api_base,
                model_name=settings.reranker_model_name,
                api_key=settings.reranker_api_key,
            )

        logger.info(f"[Retrieve Node] 发起并发重排序请求，对 {len(texts_to_rerank)} 个候选块打分...")
        scores = await reranker_client.arerank(query=query_to_rerank, texts=texts_to_rerank)
        print(scores)
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = score

        docs.sort(key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        # print(docs[0].metadata.content)
        docs = docs[:rerank_top_k]
        docs=[doc for doc in docs if doc.metadata.get("rerank_score") >= THRESHOLD_YES]
        
        if not docs:

            return {
                "retrieval_grade": "no",
                "documents": []
            }

        top_score = docs[0].metadata.get("rerank_score", 0.0) if docs else 0.0
        
        logger.info(f"[Reranker Node] 重排完成。截断保留前 {rerank_top_k} 个，Top 1 得分: {top_score:.4f}, 保留 {len(docs)} 个文档")

        logger.info("[Grader Node] 开始使用 LLM 评估检索到的文档是否有用...")
        llm = LLMManager.get_llm()
        structured_llm = llm.with_structured_output(GradeDocuments)
        prompt = ChatPromptTemplate.from_messages([
            ("system", GRADER_SYSTEM_PROMPT),
            ("human", "Retrieved documents: \n\n {documents} \n\n User question: {question}")
        ])
        retrieval_grader = prompt | structured_llm
        
        docs_text = "\n\n".join([doc.page_content for doc in docs])
        result = await retrieval_grader.ainvoke({"question": question, "documents": docs_text})
        
        grade = result.binary_score.lower()
        logger.info(f"[Grader Node] LLM 评估结果: {grade}")
        
        return {
            "retrieval_grade": grade,
            "documents": docs,
        }

    except Exception as e:
        logger.error(f"[Grader Node] 重排序或评估失败: {e}")

    return {
        "retrieval_grade": "yes",
        "documents": docs,
    }
