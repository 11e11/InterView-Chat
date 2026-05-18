# chatchat-server/chatchat/server/agent/tools_factory/interview_question_tool.py

from langchain_core.tools import BaseTool, tool
from chatchat.server.utils import build_logger, get_tool

logger = build_logger()


@tool
async def search_interview_questions(
    job_type: str,
    weak_points: str,
    question_type: str = "technical",
    kb_name: str = "samples",
) -> str:
    """
    根据候选人薄弱点和目标岗位，从知识库中召回面试题目。
    在获得简历打分结果后调用此工具出题。

    Args:
        job_type: 目标岗位，如"深度学习算法工程师"或"后端开发工程师"
        weak_points: 候选人的薄弱点（逗号分隔），来自打分结果的interview_focus字段
        question_type: 题目类型，"technical"表示技术题，"star"表示STAR行为题
        kb_name: 知识库名称，默认samples，改为你实际的知识库名
    """
    # 直接复用项目内置的 search_local_knowledgebase 工具
    search_local_knowledgebase = get_tool(name="search_local_knowledgebase")

    if question_type == "star":
        query = f"STAR行为面试题 {job_type} {weak_points} 情境 任务 行动 结果"
    else:
        query = f"{job_type} 技术面试题 {weak_points} 追问 考察意图 标准答案"

    try:
        result = await search_local_knowledgebase.ainvoke({
            "query": query,
            "knowledge_base_name": kb_name,
            "top_k": 5,
            "score_threshold": 0.3,
        })
        return result if result else f"知识库[{kb_name}]中未找到相关题目，请确认知识库已正确建立。"
    except Exception as e:
        logger.error(f"interview question tool error: {e}")
        return f"知识库检索失败：{str(e)}"