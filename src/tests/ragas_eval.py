import os
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextRecall
from langchain_openai import ChatOpenAI
from ragas.run_config import RunConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("致命错误: 找不到 OPENAI_API_KEY。请检查 .env 文件！")
    exit(1)

def main():
    # 1. 加载本地数据并转为 HuggingFace Dataset
    json_path = os.path.join(os.path.dirname(__file__), "ragas_eval_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # 强制类型清洗，防止 Ragas 底层解析崩溃
        item["contexts"] = item.get("contexts", []) if isinstance(item.get("contexts"), list) else [str(item.get("contexts"))]

    dataset = Dataset.from_list(data)

    # 2. 初始化 Langchain LLM (针对旧版指标最稳定的传入方式)
    judge_model = "glm-5.1"
    api_base = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    print(f"正在初始化裁判模型 [{judge_model}]...")
    eval_llm = ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=judge_model,
        temperature=0.0,
        timeout=None,
        max_retries=5
    )

    ragas_run_config = RunConfig(
        timeout=240,        # 留出充足的单次推理时间
        max_retries=3,      # 将默认的 10 次重试降到 3 次，不行就直接报错，防止卡死
        max_wait=15         # 将重试最长等待时间从 60 秒降到 15 秒
    )
    
    print("开始评估，这可能需要一点时间，请稍候...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ContextRecall()],
        llm=eval_llm,
        run_config=ragas_run_config,
    )

    print("\n=== 最终平均评估结果 ===")
    print(result)

    print("\n正在生成详细成绩单和可视化图表...")

    # 1. 将结果转换为 DataFrame（包含每一道题的 prompt、answer 和各项具体得分）
    df = result.to_pandas()

    # 2. 导出为 CSV，你可以直接用 Excel 打开，逐行查看哪道题得了 0 分
    csv_path = os.path.join(os.path.dirname(__file__), "ragas_detailed_scores.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"详细成绩单已保存至: {csv_path}")

    # 3. 绘制柱状图
    # 设置支持中文字体（防止图表里的中文变成方块）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 我们要画的指标列
    metrics_cols = ['faithfulness', 'context_recall']

    # 设置图表大小
    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制分组柱状图
    df[metrics_cols].plot(kind='bar', ax=ax, width=0.8, color=['#4C72B0', '#55A868'])

    ax.set_title('Ragas 每道测试题的详细得分 (Faithfulness & Context Recall)', fontsize=16, pad=15)
    ax.set_xlabel('测试题序号 (Index)', fontsize=12)
    ax.set_ylabel('得分 (0.0 - 1.0)', fontsize=12)

    # Y轴范围固定在 0 到 1.1（留点顶部空间给图例）
    ax.set_ylim(0, 1.15)
    ax.legend(['忠实度 (Faithfulness)', '上下文召回率 (Context Recall)'], loc='upper center', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 优化X轴标签，让它不那么拥挤
    plt.xticks(rotation=0)
    plt.tight_layout()

    # 4. 保存图表为 PNG 图片
    plot_path = os.path.join(os.path.dirname(__file__), "ragas_scores_chart.png")
    plt.savefig(plot_path, dpi=300)
    print(f"详细柱状图已保存至: {plot_path}")

if __name__ == "__main__":
    main()