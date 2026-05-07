import logging
import sys
import json
import asyncio
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from src.agent.state import GraphState

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters 

logger = logging.getLogger(__name__)

async def _search_single_query(session: ClientSession, query: str) -> List[Document]:
    """处理单个查询的内部辅助函数，包含 JSON 提取逻辑"""
    docs = []
    try:
        result = await session.call_tool(
            name="bing_search",
            arguments={"query": query, "count": 5}
        )
        if result.content:
            for item in result.content:
                if getattr(item, "type", "") == "text":
                    try:
                        search_data = json.loads(item.text)
                        results = search_data.get("results", [])
                        for res in results:
                            content = res.get("snippet", "") + "\n" + res.get("name", "")
                            if content.strip():
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": res.get("url", "local_mcp_bing"),
                                        "query": query,
                                        "modality": "prose"
                                    }
                                )
                                docs.append(doc)
                    except json.JSONDecodeError:
                        
                        doc = Document(
                            page_content=item.text,
                            metadata={
                                "source": "local_mcp_bing",
                                "query": query,
                                "modality": "prose"
                            }
                        )
                        docs.append(doc)
    except Exception as e:
        logger.error(f"[WebSearch Node] 单个查询失败 '{query}': {e}")
    
    return docs


async def web_search_node(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("--- 执行本地 MCP Web Search ---")

    queries = state.get("search_queries", [])
    if not queries:
        queries = [state.get("question", "")]

    queries = [q for q in queries if q and q.strip()]

    matched_domain = state.get("matched_domain", "")
    if matched_domain:
        queries = [f"{matched_domain}中: {q}" for q in queries]
        logger.info(f"-> 已为搜索关键词添加领域前缀 '{matched_domain}中: '")

    if not queries:
        logger.warning("[WebSearch Node] 没有可用的查询词。")
        return {"documents": [], "web_search_executed": True}

    logger.info(f"-> 正在通过本地 MCP 并发搜索 {len(queries)} 个查询词...")
    web_docs: List[Document] = []
    
    cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    server_params = StdioServerParameters(
        command=cmd,
        args=["-y", "bing-cn-mcp"]
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                
                tasks = [_search_single_query(session, q) for q in queries]
                
                
                results_list = await asyncio.gather(*tasks)
                
                
                for docs in results_list:
                    web_docs.extend(docs)

        logger.info(f"-> MCP 并发搜索完成，共抓取到 {len(web_docs)} 篇文档。")

    except Exception as e:
        logger.error(f"[WebSearch Node] MCP 本地调用初始化失败: {e}", exc_info=True)
        return {"documents": [], "web_search_executed": True}

    
    return {"documents": web_docs, "web_search_executed": True, "web_search_loop_step": 1}