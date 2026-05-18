# chatchat-server/chatchat/server/agent/graphs_factory/resume_scoring_graph.py

import json
import re
from typing import List, Dict, Literal

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, filter_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists
from chatchat.settings import Settings
from .graphs_registry import State, Graph, register_graph

logger = build_logger()


# ── State ─────────────────────────────────────────────────────────────────────

class ResumeScoringState(State):
    """
    简历打分 Graph 的 State，继承基础 State。
    新增字段用于存储打分过程中的中间数据。
    """
    knowledge_base: str
    top_k: int
    score_threshold: float
    question: str           # 当前轮的问题（简历文本）
    docs: List[Dict]        # 召回的知识库文档


# ── System Prompt ─────────────────────────────────────────────────────────────

SCORING_SYSTEM_PROMPT = """你是一位资深技术面试官，负责对候选人简历进行结构化打分。

工作流程：
1. 用户提供简历文本和目标岗位
2. 你必须调用知识库查询工具，检索该岗位对应的JD画像和打分SOP
   查询词示例："深度学习算法工程师 打分SOP 评分维度 技术栈要求"
3. 根据检索到的SOP对简历打分
4. 严格按照以下JSON格式输出，JSON前后不要有任何其他文字：

{
  "total_score": 75,
  "dimensions": {
    "education": {"score": 12, "reason": "985高校硕士，计算机专业"},
    "tech_stack": {"score": 28, "reason": "PyTorch熟练，缺乏TensorRT经验"},
    "project_quality": {"score": 22, "reason": "项目规模较大但量化结果不清晰"},
    "achievement": {"score": 13, "reason": "无论文，有Kaggle银牌"}
  },
  "conclusion": "推荐",
  "strengths": ["PyTorch熟练", "项目规模大"],
  "risks": ["缺乏工程部署经验"],
  "interview_focus": ["模型部署与优化", "项目量化效果追问"]
}

conclusion 只能是：强烈推荐 / 推荐 / 待定 / 拒绝
"""


def _try_parse_score_json(text: str) -> dict:
    """从LLM输出中提取JSON，容忍markdown代码块包裹"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {"error": "JSON解析失败", "raw_output": text[:500]}


# ── Graph Class ───────────────────────────────────────────────────────────────

@register_graph
class ResumeScoringGraph(Graph):
    name = "resume_scoring"
    label = "agent"        # 归类到 agent 标签，出现在图1"选择工作流"下拉框
    title = "简历打分Agent"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 knowledge_base: str = "samples",
                 top_k: int = 5,
                 score_threshold: float = 0.3):
        super().__init__(llm, tools, history_len, checkpoint)
        # 注入 search_local_knowledgebase 工具（项目内置，直接复用）
        search_local_knowledgebase = get_tool(name="search_local_knowledgebase")
        self.tools = add_tools_if_not_exists(
            tools_provides=self.tools,
            tools_need_append=[search_local_knowledgebase]
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def async_history_manager(self, state: ResumeScoringState) -> ResumeScoringState:
        """初始化 state，过滤历史中的 ToolMessage，保留干净的对话历史"""
        try:
            filtered_messages = []
            for message in filter_messages(state["messages"], exclude_types=[ToolMessage]):
                if isinstance(message, AIMessage) and message.tool_calls:
                    continue
                filtered_messages.append(message)
            state["history"] = filtered_messages[-self.history_len:]
            state["question"] = state["history"][-1].content
            state["knowledge_base"] = self.knowledge_base
            state["top_k"] = self.top_k
            state["score_threshold"] = self.score_threshold
            return state
        except Exception as e:
            raise Exception(f"Filtering messages error: {e}")

    async def chatbot(self, state: ResumeScoringState) -> ResumeScoringState:
        """主推理节点：LLM决定是否调用知识库工具，或直接输出评分"""
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        prompt = PromptTemplate(
            template="""
{system_prompt}

知识库参数：
- knowledge_base: {knowledge_base}
- top_k: {top_k}
- score_threshold: {score_threshold}

历史消息与当前简历：
{history}
""",
            input_variables=["system_prompt", "history", "knowledge_base", "top_k", "score_threshold"],
        )

        llm_chain = prompt | self.llm_with_tools
        message = await llm_chain.ainvoke({
            "system_prompt": SCORING_SYSTEM_PROMPT,
            "history": state["history"],
            "knowledge_base": state["knowledge_base"],
            "top_k": state["top_k"],
            "score_threshold": state["score_threshold"],
        })

        state["messages"] = [message]
        state["history"].append(message)
        return state

    async def format_score_output(self, state: ResumeScoringState) -> ResumeScoringState:
        """
        格式化节点：在 LLM 输出后，将 JSON 转换为可读 Markdown 报告。
        此节点只在 tools_condition 判断为 END（不再调用工具）时执行。
        """
        last_content = state["messages"][-1].content
        score_data = _try_parse_score_json(last_content)

        if "error" not in score_data:
            dims = score_data.get("dimensions", {})
            report = f"""## 📊 简历评分报告

**综合得分：{score_data.get('total_score', 'N/A')} / 100**　　**评估结论：{score_data.get('conclusion', 'N/A')}**

| 评估维度 | 得分 | 说明 |
|---------|------|------|
| 学历背景 | {dims.get('education', {}).get('score', '-')} / 15 | {dims.get('education', {}).get('reason', '-')} |
| 技术栈匹配 | {dims.get('tech_stack', {}).get('score', '-')} / 35 | {dims.get('tech_stack', {}).get('reason', '-')} |
| 项目质量 | {dims.get('project_quality', {}).get('score', '-')} / 30 | {dims.get('project_quality', {}).get('reason', '-')} |
| 成果影响力 | {dims.get('achievement', {}).get('score', '-')} / 20 | {dims.get('achievement', {}).get('reason', '-')} |

**核心优势：** {', '.join(score_data.get('strengths', []))}
**主要风险：** {', '.join(score_data.get('risks', []))}
**面试追问重点：** {', '.join(score_data.get('interview_focus', []))}

---
*切换到「面试模拟Agent」工作流可进行模拟面试。*
"""
            state["messages"].append(AIMessage(content=report))
        # 解析失败则保留原始输出，不追加新消息

        return state

    def get_graph(self) -> CompiledStateGraph:
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")

        graph_builder = StateGraph(ResumeScoringState)
        retrieve = ToolNode(tools=self.tools)

        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", retrieve)
        graph_builder.add_node("format_output", self.format_score_output)

        graph_builder.add_edge(START, "history_manager")
        graph_builder.add_edge("history_manager", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "tools",  # 还需要检索 → 去 tools
                END: "format_output",  # LLM认为完成 → 格式化输出
            },
        )
        graph_builder.add_edge("tools", "chatbot")  # 检索结果返回 → 继续推理
        graph_builder.add_edge("format_output", END)

        return graph_builder.compile(checkpointer=self.checkpoint)

    @staticmethod
    def handle_event(node: str, event: ResumeScoringState) -> BaseMessage:
        return event["messages"][-1]