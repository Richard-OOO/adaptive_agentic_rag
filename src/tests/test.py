import json
import logging
import sys
import asyncio
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

LOG_FILE = "eval_process.log"
OUTPUT_JSON_FILE = "ragas_eval_results.json"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("EvalLogger")
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setFormatter(formatter)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


async def run_eval():
    from src.agent.graph import build_adaptive_rag_graph
    workflow = build_adaptive_rag_graph()
    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("[Eval] MemorySaver 模式已就绪")

    current_dir = Path(__file__).parent
    json_path = current_dir / "test.json"

    if not json_path.exists():
        logger.error(f" 未找到输入文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    logger.info(f" 成功加载 {len(test_cases)} 条测试用例，开始执行自动化评估...")

    ragas_results = []
    stats = {
        "total": len(test_cases),
        "retrieval_success": 0,
        "no_hallucination": 0,
        "errors": 0,
    }

    for i, tc in enumerate(test_cases, start=1):
        question = tc.get("question", "")
        ground_truth = tc.get("ground_truth", "")
        history_data = tc.get("history", [])
        messages = []
        for msg in history_data:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        logger.info("=" * 70)
        logger.info(f" 测试 Case [{i}/{len(test_cases)}]: {question}")

        inputs = {"question": question, "messages": messages}
        config = {
            "configurable": {
                "thread_id": f"eval_thread_{i}",
                "user_id": "admin",
                "session_id": "init_session",
                "search_top_k": 5,
                "rerank_top_k": 3,
                "knowledge_domains": ["心理学"],
            }
        }

        final_state = {}

        try:
            async for event in app.astream(inputs, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    
                    final_state.update(state_update)


            answer = final_state.get("generation", "未能生成答案")

            raw_docs = final_state.get("documents", [])
            contexts = [doc.page_content for doc in raw_docs] if raw_docs else []

            ragas_row = {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth
            }
            ragas_results.append(ragas_row)

            r_grade = final_state.get("retrieval_grade", "no")
            potential_hallucination = final_state.get("potential_hallucination", False)

            if r_grade == "yes":
                stats["retrieval_success"] += 1
            if not potential_hallucination:
                stats["no_hallucination"] += 1

            logger.info(f" Case [{i}] 完成 | 检索: {r_grade}, 潜在幻觉: {potential_hallucination}")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f" 测试 Case [{i}] 发生异常: {e}", exc_info=True)

    out_path = current_dir / OUTPUT_JSON_FILE
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_results, f, ensure_ascii=False, indent=2)

    valid = stats["total"] - stats["errors"]
    logger.info("\n" + "=" * 50)
    logger.info("内部评估节点统计面板 (Internal Graph Stats)")
    logger.info("=" * 50)
    logger.info(f"  总测试题数         : {stats['total']}")
    logger.info(f"  执行错误数         : {stats['errors']}")
    logger.info(f"  检索成功率         : {stats['retrieval_success']}/{valid} ({(stats['retrieval_success']/max(1, valid))*100:.1f}%)")
    logger.info(f"  无幻觉率           : {stats['no_hallucination']}/{valid} ({(stats['no_hallucination']/max(1, valid))*100:.1f}%)")
    logger.info("=" * 50)
    logger.info(f" 评测数据已保存至: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())
