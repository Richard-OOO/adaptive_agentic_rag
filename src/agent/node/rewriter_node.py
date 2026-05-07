import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from src.core.llm_manager import LLMManager

from src.agent.state import GraphState
from langchain_core.documents import Document
from langchain_core.messages import trim_messages

logger = logging.getLogger(__name__)


class RewriteDecision(BaseModel):
    question: str = Field(description="将用户的问题补全代词含义后的问题，用于重排打分，不用拆解多跳")
    search_queries: List[str] = Field(description="重写后的查询列表，需要拆解多跳，需补全代词，用于检索")


REWRITER_SYSTEM_PROMPT = """你是一个查询重写与纠错器以及降噪优化器（Rewriter）
当检索尚未执行（retrieval_grade 为 "none"）时，你应执行事前优化：
  - 进行代词指代消解，结合对话历史补全主语；
  如将"它"在上下文中代指的"某个概念"补全到查询中,并且更新用户的原始问题，为其补充代词；
  - 将复杂/多跳问题拆分为多个独立查询；例如:"
  法国和中国的国土面积哪个更大？" 可以拆分为 "法国的国土面积是多少？" 和 "中国的国土面积是多少？" 两个查询；
  -进行搜索词降噪优化，只跟据标签相关的方向优化查询词，其他无关的元素可以过滤掉，只保留标签对应的元素
    
  
当检索后评估为 "no" 或 "partial" 时，你应执行事后纠错：
 事后纠错反思阶段（有检索反馈）：
- 阅读【当前检索反馈】中的召回片段，判断为何偏离主题；
- 阅读【历史尝试查询】，绝对禁止生成与历史相同的查询词；
- 提取底层近义词，或转换视角进行重写。
请只返回符合 Schema 的 JSON，字段为 `search_queries`（一个字符串数组），`question`（一个字符串）。不是markdown格式的json字符串"""
async def rewriter_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    question = state.get("question")
    if not question:
        raise ValueError("[Rewriter Node] state 缺失 question")

    messages = state.get("messages", [])
    retrieval_grade = state.get("retrieval_grade", "none")
    retrieval_reasoning = state.get("retrieval_reasoning", "无检索反馈")
    past_search_queries = state.get("past_search_queries", [])
    tag = state.get("matched_domain", [])
    configurable = config.get("configurable", {})
    llm = LLMManager.get_llm(
        
    )
    structured = llm.with_structured_output(RewriteDecision)

    human_template = (
        "【原始问题】: {question}\n"
        "【当前检索反馈】: {retrieval_reasoning}\n"
        "【历史尝试查询(请勿重复)】: {past_search_queries}\n\n"
        "【当前搜索领域标签】: {tag}\n"
        "请返回优化后的 search_queries 列表。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("human", human_template)
    ])

    chain = prompt | structured

    trimmed_messages = trim_messages(
        messages,
        max_tokens=50,
        strategy="last",
        token_counter=len,
        include_system=True,
    )

    try:
        decision: RewriteDecision = await chain.ainvoke({
            "question": question, 
            "messages": trimmed_messages, 
            "retrieval_reasoning": retrieval_reasoning,
            "past_search_queries": ", ".join(past_search_queries) if past_search_queries else "无",
            "tag": tag if tag else "无"
        })
        return {
            "question": decision.question,
            "search_queries": decision.search_queries, 
            "past_search_queries": decision.search_queries if decision.search_queries else [],
        }
    except Exception as e:
        logger.error(f"[Rewriter Node] 重写失败: {e}")
        
        return {"search_queries": [question]}
