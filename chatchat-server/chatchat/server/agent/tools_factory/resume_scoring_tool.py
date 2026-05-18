from langchain_core.tools import tool
from chatchat.server.agent.tools_factory.tools_registry import regist_tool

@regist_tool(title="简历打分工具")
@tool
async def resume_scoring(resume_text: str, job_type: str) -> str:
    """
    根据简历文本和目标岗位进行结构化打分。
    Args:
        resume_text: 候选人简历的完整文本
        job_type: 目标岗位类型，如"深度学习算法工程师"、"后端开发工程师"
    """
    # 从知识库召回对应岗位的JD画像和打分SOP
    from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
    kb = KBServiceFactory.get_service("interview_kb", "faiss")
    docs = await kb.asearch_docs(f"{job_type} 打分SOP 评分维度", top_k=5)
    context = "\n".join([d.page_content for d in docs])
    
    # 返回context供上层LLM生成结构化JSON评分
    return f"【岗位SOP上下文】\n{context}\n\n【待评简历】\n{resume_text}"