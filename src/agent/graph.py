import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from src.agent.state import GraphState
from src.agent.node.router import intent_router_node
from src.agent.node.rewriter_node import rewriter_node
from src.agent.node.retrieve import retrieve_node
from src.agent.node.grader_node import grader_node
from src.agent.node.generate_node import generate_node
from src.agent.node.web_search_node import web_search_node

logger = logging.getLogger(__name__)
REGENERATE_MAX_LOOPS = 2
MAX_LOOPS = 3


class AgentConfig(TypedDict, total=False):
    user_id: str
    session_id: str
    trace_id: str
    search_top_k: Optional[int]
    rerank_top_k: Optional[int]
    enable_web_search_fallback: bool
    knowledge_domains: list[str]


def route_after_router(state: GraphState) -> Literal["generate", "rewrite", "retrieve", "web_search"]:
    logger.info("--- ROUTE AFTER ROUTER ---")

    if not state.get("need_retrieval", True):
        logger.info("-> Bypass: entering generate node.")
        return "generate"

    if not state.get("matched_domain"):
        logger.info("-> No matched domain: entering web search node.")
        return "web_search"

    if state.get("need_query_optimization", False):
        logger.info("-> Complex/multi-hop query: entering rewriter node.")
        return "rewrite"

    logger.info("-> Standard query: entering retrieval node.")
    return "retrieve"


def route_after_grader(state: GraphState) -> Literal["generate", "rewrite"]:
    logger.info("--- ROUTE AFTER GRADER ---")
    web_search_loop_step = state.get("web_search_loop_step", 0)
    grade = state.get("retrieval_grade", "no")
    if grade == "yes":
        logger.info("-> Retrieval quality sufficient: entering generate node.")
        return "generate"
    if web_search_loop_step >= MAX_LOOPS:
        logger.warning(f"-> Max web search loops ({MAX_LOOPS}) reached: forcing generate.")
        return "generate"
    return "rewrite"


def route_after_rewrite(state: GraphState) -> Literal["retrieve", "web_search", "generator"]:
    logger.info("--- ROUTE AFTER REWRITE ---")
    retrieve_loop_step = state.get("retrieve_loop_step", 0)
    search_web_step = state.get("web_search_loop_step", 0)
    if retrieve_loop_step >= MAX_LOOPS:
        logger.warning(f"-> Max retrieval loops ({MAX_LOOPS}) reached.")
        if search_web_step >= MAX_LOOPS:
            logger.warning(f"-> Max web search loops ({MAX_LOOPS}) reached: forcing generate.")
            return "generator"
        return "web_search"
    return "retrieve"


def build_adaptive_rag_graph() -> StateGraph:
    workflow = StateGraph(GraphState, config_schema=AgentConfig)

    workflow.add_node("router", intent_router_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("generator", generate_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "generate": "generator",
            "web_search": "web_search",
            "rewrite": "rewriter",
            "retrieve": "retriever",
        }
    )

    workflow.add_conditional_edges(
        "rewriter",
        route_after_rewrite,
        {
            "retrieve": "retriever",
            "web_search": "web_search",
            "generator": "generator",
        }
    )

    workflow.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "generate": "generator",
            "rewrite": "rewriter",
        }
    )
    workflow.add_edge("retriever", "grader")
    workflow.add_edge("web_search", "grader")
    workflow.add_edge("generator", END)

    return workflow


async def build_compiled_app(checkpointer=None):
    workflow = build_adaptive_rag_graph()

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)

    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from src.core.db_client import MongoDBClientManager

        sync_mongo_client = MongoDBClientManager.get_client(sync=True)
        checkpointer = MongoDBSaver(sync_mongo_client)
        logger.info("[Graph] MongoDBSaver checkpoint 初始化成功")
        return workflow.compile(checkpointer=checkpointer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.warning(f"[Graph] MongoDBSaver 初始化失败 ({e})，尝试 Redis 降级...")

        try:
            from langgraph.checkpoint.redis import RedisSaver
            import redis
            from src.core.config import get_settings

            settings = get_settings()
            sync_redis = redis.from_url(settings.redis_url, decode_responses=False)
            sync_redis.ping()
            checkpointer = RedisSaver(redis_client=sync_redis)
            checkpointer.setup()
            logger.info("[Graph] RedisSaver checkpoint 初始化成功 (降级)")
            return workflow.compile(checkpointer=checkpointer)
        except Exception as e2:
            logger.warning(f"[Graph] RedisSaver 初始化也失败 ({e2})，使用 MemorySaver 内存模式")
            from langgraph.checkpoint.memory import MemorySaver
            return workflow.compile(checkpointer=MemorySaver())
if __name__ == "__main__":
    import asyncio
    import json
    
    def custom_serializer(obj):
        """兼容 LangChain/Pydantic 对象的简单序列化器"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    async def run_local_test():
        print("========================================")
        print("启动 LangGraph 单步调试模式 (MemorySaver)")
        print("========================================")

        # 1. 强制使用 MemorySaver 编译图，避免测试时污染 Redis 或 MongoDB
        from langgraph.checkpoint.memory import MemorySaver
        workflow = build_adaptive_rag_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        # 2. 构造测试输入
        messages_data = {

    "history": [
      {"role": "user", "content": "请介绍一下格式塔心理学。"},
      {"role": "assistant", "content": "格式塔心理学的代表人物是韦特海默和苛勒。"}
    ],
    "question": "它的核心观点用一句话概括是什么？",
    }
        messages = []
        from langchain_core.messages import HumanMessage, AIMessage
        for msg in messages_data["history"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        inputs = {
            "question": "世界上第一个心理学实验室",
            "messages": messages
        }
        config = {
            "configurable": {
                "thread_id": "debug_memory_thread_01",
                "user_id": "test_user",
                "session_id": "test_session",
                "search_top_k": 3,
                "rerank_top_k": 2,
                "knowledge_domains": ["心理学"]
            }
        }

        print(f"\n[测试输入]: {inputs['question']}")
        print("提示: 节点执行完毕后系统会暂停，按【回车键/空格+回车】继续，输入 'q' 退出。\n")

        try:
            # 3. 使用 stream_mode="updates" 捕获每个节点的状态增量
            async for event in app.astream(inputs, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    print(f"\n 节点执行完成: 【{node_name}】")
                    
                    # 4. 简单清理控制台输出，防止 Document 或 Message 列表刷屏
                    clean_update = {}
                    for k, v in state_update.items():
                        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'page_content'):
                            clean_update[k] = f"[List of {len(v)} Documents] 预览: {v[0].page_content[:30]}..."
                        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'content'):
                            clean_update[k] = f"[List of {len(v)} Messages]"
                        else:
                            clean_update[k] = v

                    print("状态增量:")
                    print(json.dumps(clean_update, ensure_ascii=False, indent=2, default=custom_serializer))
                    print("-" * 50)
                    
                    # 5. 阻断式等待用户输入 (回车或空格均可，只要按回车确认)
                    cmd = input(" 按【回车】执行下一个节点，或输入 'q' 退出: ")
                    if cmd.strip().lower() == 'q':
                        print(" 收到退出指令，测试终止。")
                        return

            print("\n图流转完全结束！最终流程已跑完。")
            
        except Exception as e:
            print(f"\n 图流转发生异常: {e}")
            logger.error(f"详细报错信息:", exc_info=True)

    # 执行异步主函数
    asyncio.run(run_local_test())