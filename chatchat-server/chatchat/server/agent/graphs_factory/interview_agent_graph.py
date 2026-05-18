# chatchat-server/chatchat/server/agent/graphs_factory/interview_agent_graph.py

import json
import re
from typing import List, Dict, Literal, Union

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, filter_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists
from .graphs_registry import State, Graph, register_graph

logger = build_logger()


# ── 工具定义（LLM自主决策调用）────────────────────────────────────────────────

@tool
async def search_knowledge(query: str, kb_name: str = "samples") -> str:
    """
    检索本地知识库。用于查找：岗位JD、打分SOP、技术面试题、STAR行为题。
    
    Args:
        query: 检索词，如"深度学习工程师打分SOP"或"注意力机制面试题追问"
        kb_name: 知识库名称
    """
    search_local_knowledgebase = get_tool(name="search_local_knowledgebase")
    try:
        result = await search_local_knowledgebase.ainvoke({
            "query": query,
            "database": kb_name,
            "top_k": 5,
            "score_threshold": 0.3,
        })
        return result or "知识库中未找到相关内容"
    except Exception as e:
        logger.error(f"search_knowledge error: {e}")
        return f"检索失败: {str(e)}"


# @tool
# def parse_resume_score(llm_score_text: Union[str, dict]) -> str:
#     """
#     将LLM输出的评分JSON格式化为Markdown报告直接返回给用户。
#     当LLM完成简历分析后必须调用此工具。

#     Args:
#         llm_score_text: LLM生成的评分JSON（字符串或dict均可）
#     """
#     def _extract_json(text) -> dict:
#         if isinstance(text, dict):
#             return text
#         try:
#             return json.loads(text.strip())
#         except json.JSONDecodeError:
#             pass
#         for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
#             match = re.search(pattern, text, re.DOTALL)
#             if match:
#                 try:
#                     return json.loads(match.group(1))
#                 except json.JSONDecodeError:
#                     continue
#         return {}

#     # 兼容 {"score": x, "reason/comment": y} 和直接是数字两种结构
#     def get_score(val):
#         if isinstance(val, dict):
#             return val.get("score", "-")
#         return val if val != "-" else "-"

#     def get_reason(val):
#         if isinstance(val, dict):
#             return val.get("reason") or val.get("comment", "-")
#         return "-"

#     # 兼容字符串和列表
#     def to_str(val):
#         if isinstance(val, list):
#             return "、".join(str(v) for v in val)
#         return str(val) if val else "-"

#     score_data = _extract_json(llm_score_text)
#     if not score_data:
#         return f"[评分解析失败，原始内容：{str(llm_score_text)[:200]}]"

#     dims = score_data.get("dimensions", {})

#     report = f"""## 📊 简历评分报告

# **综合得分：{score_data.get('total_score', 'N/A')} / 100**

# | 维度 | 得分 | 说明 |
# |------|------|------|
# | 学历背景 | {get_score(dims.get('education', '-'))} | {get_reason(dims.get('education', {}))} |
# | 技术栈 | {get_score(dims.get('tech_stack', '-'))} | {get_reason(dims.get('tech_stack', {}))} |
# | 项目质量 | {get_score(dims.get('project_quality', '-'))} | {get_reason(dims.get('project_quality', {}))} |
# | 成果影响力 | {get_score(dims.get('achievement', '-'))} | {get_reason(dims.get('achievement', {}))} |

# **综合结论：** {score_data.get('conclusion', '-')}

# **核心优势：** {to_str(score_data.get('strengths', '-'))}

# **主要风险：** {to_str(score_data.get('risks', '-'))}

# **面试追问重点：** {to_str(score_data.get('interview_focus', '-'))}

# ---
# ✅ 评分完成！请告诉我你想模拟哪个方向的面试，我将开始第1/5轮提问。
# """
#     return report  # 返回字符串而不是dict，ToolMessage可以直接显示


@tool
def evaluate_answer(question: str, candidate_answer: str, job_type: str) -> str:
    """
    评估候选人对某道面试题的回答质量，返回格式化的评分结果。
    LLM在候选人作答后调用此工具，然后将评分结果展示给用户并出下一题。

    Args:
        question: 面试官提出的题目
        candidate_answer: 候选人的回答内容
        job_type: 目标岗位
    """
    answer_len = len(candidate_answer)
    has_example = any(kw in candidate_answer for kw in ["例如", "比如", "项目中", "实际上", "曾经"])
    has_numbers = any(char.isdigit() for char in candidate_answer)

    # 基于规则给出初步分数供参考
    base_score = 5
    if answer_len > 200:
        base_score += 1
    if has_example:
        base_score += 2
    if has_numbers:
        base_score += 1
    base_score = min(base_score, 10)

    return f"""## 📝 第X轮回答评估

**题目：** {question[:50]}...

**回答分析：**
- 回答长度：{answer_len} 字（{"充分" if answer_len > 150 else "偏短"}）
- 是否有举例：{"✅ 有" if has_example else "❌ 无"}
- 是否有量化数据：{"✅ 有" if has_numbers else "❌ 无"}

**参考得分：{base_score}/10**

请你（LLM）根据以上信息，结合回答内容，输出：
1. 实际得分（0-10分）
2. 一句话点评
3. 是否追问（yes继续追问/no出下一题）
"""


@tool
def generate_interview_summary(rounds_data: str) -> str:
    """
    生成面试总结报告。在所有面试轮次完成后调用此工具。
    
    Args:
        rounds_data: 各轮面试的题目和得分信息（JSON字符串）
    """
    try:
        rounds = json.loads(rounds_data)
    except Exception:
        rounds = []

    total = sum(r.get("score", 0) for r in rounds)
    avg = total / len(rounds) if rounds else 0

    level = (
        "优秀，建议直接进入下一轮" if avg >= 8 else
        "良好，建议通过" if avg >= 6 else
        "一般，建议再考察" if avg >= 4 else
        "较弱，建议谨慎"
    )

    summary = f"""## 🎯 面试总结报告

**综合评级：{level}**（平均得分 {avg:.1f}/10）

| 轮次 | 题目摘要 | 得分 |
|------|----------|------|
"""
    for i, r in enumerate(rounds, 1):
        summary += f"| 第{i}轮 | {r.get('question', '')[:30]}... | {r.get('score', '-')}/10 |\n"

    summary += "\n**建议提升方向：** 请根据得分较低的轮次针对性准备。"
    return summary

# ── System Prompt ─────────────────────────────────────────────────────────────

INTERVIEW_AGENT_PROMPT = """你是一个智能面试助理，可以完成简历打分和模拟面试两个任务。

【你拥有以下工具】
- search_knowledge：检索知识库，获取岗位JD、打分SOP、面试题目
- evaluate_answer：评估候选人的面试回答质量（候选人作答后调用）
- generate_interview_summary：生成面试总结（所有轮次结束后调用）

【工作流程】
阶段1 - 简历打分：
  用户提交简历 → 调用 search_knowledge 检索该岗位SOP → 直接输出以下格式的Markdown评分报告：

## 📊 简历评分报告

**综合得分：XX / 100**

| 维度 | 得分 | 说明 |
|------|------|------|
| 学历背景 | XX | 说明 |
| 技术栈 | XX | 说明 |
| 项目质量 | XX | 说明 |
| 成果影响力 | XX | 说明 |

**综合结论：** ...
**核心优势：** ...
**主要风险：** ...
**面试追问重点：** ...

---
✅ 评分完成！请告诉我你想模拟哪个方向的面试，我将开始第1/5轮提问。

阶段2 - 模拟面试（用户说"开始面试"后进入）：
  → 调用 search_knowledge 检索题库 → 出题
  → 等候选人回答 → 调用 evaluate_answer 评估
  → 决定追问还是出新题
  → 5轮后调用 generate_interview_summary 输出总结

【关键规则】
- 不要替候选人作答
- 候选人作答后，必须先调用 evaluate_answer 工具，然后根据工具返回结果输出评分点评，再决定追问或出下一题
- 输出评分点评的格式：
  **【第X轮评分】得分：X/10**
  点评：...
  （追问内容 或 "进入下一题"）
- 不要跳过评分直接出下一题
- 每轮只出1道题，格式：**【第X/5轮 | 技术题/行为题】** 题目内容
- 工具调用失败时，直接根据自身知识继续
"""

# ── Graph Class ───────────────────────────────────────────────────────────────

@register_graph
class InterviewAgentGraph(Graph):
    name = "interview_agent"
    label = "agent"
    title = "简历打分+面试模拟Agent"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 knowledge_base: str = None,
                 top_k: int = None,
                 score_threshold: float = None):
        super().__init__(llm, tools, history_len, checkpoint)

        # 注入4个业务工具，LLM自主决策调用哪个
        self.tools = add_tools_if_not_exists(
            tools_provides=self.tools,
            tools_need_append=[
                search_knowledge,
                # parse_resume_score,
                evaluate_answer,
                generate_interview_summary,
            ]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", INTERVIEW_AGENT_PROMPT),
            ("placeholder", "{history}"),
        ])
        self.llm_with_tools = prompt | self.llm.bind_tools(self.tools)

    async def chatbot(self, state: State) -> State:
        """主推理节点，LLM自主决策调用哪个工具"""
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        messages = await self.llm_with_tools.ainvoke(state)
        state["messages"] = [messages]
        state["history"].append(messages)
        return state

    def get_graph(self) -> CompiledStateGraph:
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")

        graph_builder = StateGraph(State)
        tool_node = ToolNode(tools=self.tools)

        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", tool_node)

        graph_builder.set_entry_point("history_manager")
        graph_builder.add_edge("history_manager", "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")

        return graph_builder.compile(checkpointer=self.checkpoint)

    @staticmethod
    def handle_event(node: str, event: State) -> BaseMessage:
        return event["messages"][-1]