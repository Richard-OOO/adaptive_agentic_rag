import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from src.core.llm_manager import LLMManager
from langchain_core.messages import HumanMessage, trim_messages

from src.agent.state import GraphState

logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    """查询分析与路由决策结构体"""
    need_retrieval: bool = Field(
        default=True,
        description="是否需要检索外部知识库。仅当问题与知识库领域完全无关(如纯闲聊/问候)时设为 False。")

    need_query_optimization: bool = Field(
        default=True,
        description="1.如果问题包含代词(如'它')需要结合上下文重写，2.问题是一个包含多个子问题的复杂问题需要拆分，设为 True。3.如果有无关闲杂的噪音，比如说，当你认定问题的相关标签是一个垂直领域的时候，如果问题中出现了其他杂乱的内容，需要提纯核心知识点。")
    sql_filters: Dict[str, Any] = Field(
        default={},
        description="Metadata Extension Point：预留的元数据过滤条件。当前纯文本版本统一固定返回空字典 {}。为未来保留扩展位。")
    requires_extended_context: bool = Field(
        default=True,
        description="Dynamic Granularity Routing 标志：是否需要宏观上下文。当要求'总结'、'全貌'时设为 True；具体细节设为 False。")
    matched_domain: Optional[str] = Field(
        default=None,
        description="从知识库领域标签中匹配到的标签。如果问题属于知识库覆盖的某个领域，返回该标签；如果问题与所有领域都无关(如闲聊/问候/常识)，返回 None。")

ROUTER_SYSTEM_PROMPT = """你是一个高级的纯文本 RAG 查询意图分析和路由中枢。你的任务是分析用户的输入，并输出严格的 JSON 结构化决策。

核心决策逻辑与原则：

1. 【领域匹配与检索决策】
    当前知识库覆盖以下领域标签：{knowledge_domains}
    
    判断规则：
    - 如果用户的问题与上述任一领域标签相关，`need_retrieval` 必须为 True，`matched_domain` 设为最相关的那个标签。
    - 仅当问题与所有领域标签都完全无关时（如纯闲聊、问候、简单寒暄），`need_retrieval` 才设为 False，`matched_domain` 设为 None。
    - 重要：即使你认为自身可以回答该问题，只要问题属于知识库覆盖的领域，也必须检索！因为知识库中的信息可能比你的参数知识更准确、更具体、更可引用。
    - 反例：用户问"心理学是什么？"——虽然这是常识，但"心理学"属于知识库领域，必须检索。
    - 正例：用户问"你好"或"今天天气怎么样"——与知识库领域无关，need_retrieval=False。
2. 【Adaptive Query Decomposition】
    - 单一意图："什么是 LangGraph？" -> ["什么是 LangGraph？"]`need_query_optimization` 必须为 False。
    - 依赖跳跃："苹果现任CEO出生地？" -> ["苹果现任CEO是谁", "苹果现任CEO出生在哪里"]`need_query_optimization` 必须为 True。
    - 复杂问题的过滤："用Python写一段实现渐进式肌肉放松训练的代码" -> 必须检索且 need_query_optimization=True，过滤无关信息且核心是有关垂直领域标签的内容:渐进式肌肉放松）
    - 对复杂问题中无关信息的过滤，如果问题出现了除标签以为，可能会影响到查询的实体，需要将need_query_optimization设为True。
    - 正例 : 用户问"请帮我以鲁迅的口吻解释一下什么是“社会惰化”" -> 必须检索且 need_query_optimization=True，过滤无关信息且核心是有关垂直领域标签的内容:社会惰化。
    - 如果问题包含代词(如'它')需要结合上下文重写。防止查询出现问题

3. 【Dynamic Granularity Routing】
    - 独立细节查询："timeout 参数是多少？" -> `requires_extended_context`: False
    - 宏观总结查询："请总结核心内容" -> `requires_extended_context`: True

你必须直接返回符合 Schema 的 JSON，不允许任何解释。对于 sql_filters，固定返回空字典。
直接返回json字符串，不是markdown格式的json字符串。"""


async def intent_router_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    """意图路由节点"""
    question = state.get("question")
    if not question:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            question = messages[-1].content
        else:
            raise ValueError("[Router Node] State 中缺失问题(question)。")

    logger.info(f"===> [Router Node] 开始分析 Query: '{question}'")
    
    configurable = config.get("configurable", {})
    knowledge_domains = configurable.get("knowledge_domains", [])
    domains_str = "、".join(knowledge_domains) if knowledge_domains else "未指定"
    
    llm = LLMManager.get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    
    system_prompt = ROUTER_SYSTEM_PROMPT.format(knowledge_domains=domains_str)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "User Question: {question}")
    ])

    router_chain = prompt | structured_llm

    messages = state.get("messages", [])
    trimmed_messages = trim_messages(
        messages,
        max_tokens=50,
        strategy="last",
        token_counter=len,
        include_system=True,
    )

    try:
        decision: RouteDecision = await router_chain.ainvoke({"messages": trimmed_messages, "question": question})
        
        logger.info(f"[Router Node] 决策结果: need_retrieval={decision.need_retrieval}, "
                    f"need_query_optimization={decision.need_query_optimization}, "
                    f"matched_domain={decision.matched_domain}, "
                    f"filters={decision.sql_filters}, "
                    f"extended_ctx={decision.requires_extended_context}")
        
        return {
            "need_retrieval": decision.need_retrieval,
            "search_queries": [question] if decision.need_retrieval else [],
            "need_query_optimization": decision.need_query_optimization,
            "sql_filters": decision.sql_filters,
            "requires_extended_context": decision.requires_extended_context,
            "matched_domain": decision.matched_domain or "",
            "loop_step": 1
        }
        
    except Exception as e:
        logger.error(f"[Router Node] 结构化解析失败，触发 Fallback 路由: {e}")
        fallback_domain = knowledge_domains[0] if knowledge_domains else ""
        return {
            "need_retrieval": True,
            "search_queries": [question],
            "need_query_optimization": False,
            "sql_filters": {},
            "requires_extended_context": False,
            "matched_domain": fallback_domain,
            "loop_step": 1
        }
