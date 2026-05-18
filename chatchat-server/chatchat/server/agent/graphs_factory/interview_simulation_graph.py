# chatchat-server/chatchat/server/agent/graphs_factory/interview_simulation_graph.py

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists
from .graphs_registry import State, Graph, register_graph

logger = build_logger()


INTERVIEW_SYSTEM_PROMPT = """你是一位严格但专业的技术面试官，正在进行模拟面试。

规则：
1. 首次对话时，先询问候选人：目标岗位、希望题型（技术题/行为题）、薄弱点
2. 收到候选人信息后，调用 search_interview_questions 工具从题库召回题目
3. 每轮只出1道题，等候选人回答后再决定追问或出新题：
   - 回答完整 → 换新题（再次调用工具召回）
   - 回答不完整 → 追问（最多2次），不需要调用工具
4. 题目输出格式：
   **【第X轮 | 技术题/行为题】题目内容**
   *考察意图：...*
   *期望要点：...*
5. 共进行5轮后，输出面试总结（各题表现+综合建议）

注意：只扮演面试官，不替候选人作答。
"""


@register_graph
class InterviewSimulationGraph(Graph):
    name = "interview_simulation"
    label = "agent"        # 出现在图1"选择工作流"下拉框
    title = "面试模拟Agent"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 knowledge_base: str = None,
                 top_k: int = None,
                 score_threshold: float = None):
        super().__init__(llm, tools, history_len, checkpoint)

        # 注入面试出题工具
        from chatchat.server.agent.tools_factory.interview_question_tool import search_interview_questions
        self.tools = add_tools_if_not_exists(
            tools_provides=self.tools,
            tools_need_append=[search_interview_questions]
        )

        # 对齐 base_agent.py 的 prompt 构造方式
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTERVIEW_SYSTEM_PROMPT),
                ("placeholder", "{history}"),
            ]
        )
        self.llm_with_tools = prompt | self.llm.bind_tools(self.tools)

    async def chatbot(self, state: State) -> State:
        """面试主节点，对齐 base_agent.py 的 chatbot 实现"""
        # 与 base_agent.py 完全相同的 ToolMessage 处理逻辑
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

        # 节点与 base_agent.py 完全对齐
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