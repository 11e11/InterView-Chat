from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from streamlit_extras.bottom_container import bottom

from chatchat.server.agent.graphs_factory.graphs_registry import (
    list_graph_titles_by_label,
    get_graph_class_by_label_and_title
)
from chatchat.webui_pages.utils import *

from chatchat.server.utils import (
    build_logger,
    get_tool,
    list_tools,
    create_agent_models,
    get_checkpointer
)

logger = build_logger()


# @st.dialog("输入初始化内容", width="large")
# def article_generation_init_setting():
#     article_links = st.text_area("文章链接")
#     image_links = st.text_area("图片链接")
#
#     if st.button("确认"):
#         st.session_state["article_links"] = article_links
#         st.session_state["image_links"] = image_links
#         # 将 article_generation_init_break_point 状态扭转为 True, 后续将进行 update_state 动作
#         st.session_state["article_generation_init_break_point"] = True
#
#         user_input = (f"文章链接: {article_links}\n"
#                       f"图片链接: {image_links}")
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         st.session_state.messages.append({
#             "role": "user",
#             "content": user_input,
#             "type": "text"  # 标识为文本类型
#         })
#
#         st.rerun()


# @st.dialog("开始改写文章", width="large")
# def article_generation_start_setting():
#     cols = st.columns(3)
#     platforms = ["所有"] + list(get_config_platforms())
#     platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
#     llm_models = list(
#         get_config_models(
#             model_type="llm", platform_name=None if platform == "所有" else platform
#         )
#     )
#     llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
#     temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
#     with st.container(height=300):
#         st.markdown(st.session_state["article_list"])
#     prompt = st.text_area("指令(Prompt):", value="1.将上述提供的文章内容列表,各自提炼出提纲;\n"
#                                                  "2.将提纲列表整合成一篇文章的提纲;\n"
#                                                  "3.按照整合后的提纲, 生成一篇新的文章, 字数要求 500字左右;\n"
#                                                  "4.只需要返回最后的文章内容即可.")
#
#     if st.button("开始编写"):
#         st.session_state["platform"] = platform
#         st.session_state["llm_model"] = llm_model
#         st.session_state["temperature"] = temperature
#         st.session_state["prompt"] = prompt
#         # 将 article_generation_start_break_point 状态扭转为 True, 后续将进行 update_state 动作
#         st.session_state["article_generation_start_break_point"] = True
#
#         user_input = (f"模型: {llm_model}\n"
#                       f"温度: {temperature}\n"
#                       f"指令: {prompt}")
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         st.session_state.messages.append({
#             "role": "user",
#             "content": user_input,
#             "type": "text"  # 标识为文本类型
#         })
#
#         st.rerun()


# @st.dialog("文章重写确认", width="large")
# def article_generation_repeat_setting():
#     cols = st.columns(3)
#     platforms = ["所有"] + list(get_config_platforms())
#     platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
#     llm_models = list(
#         get_config_models(
#             model_type="llm", platform_name=None if platform == "所有" else platform
#         )
#     )
#     llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
#     temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
#     with st.container(height=300):
#         st.markdown(st.session_state["article"])
#     prompt = st.text_area("指令(Prompt):", value="请继续优化, 最后只需要返回文章内容.")
#
#     if st.button("确认-需要重写"):
#         st.session_state["platform"] = platform
#         st.session_state["llm_model"] = llm_model
#         st.session_state["temperature"] = temperature
#         st.session_state["prompt"] = prompt
#         st.session_state["article_generation_repeat_break_point"] = True
#
#         user_input = (f"模型: {llm_model}\n"
#                       f"温度: {temperature}\n"
#                       f"指令: {prompt}")
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         st.session_state.messages.append({
#             "role": "user",
#             "content": user_input,
#             "type": "text"  # 标识为文本类型
#         })
#         st.rerun()
#     if st.button("确认-不需要重写"):
#         # 如果不需要继续改写, 则固定 prompt 如下
#         prompt = "不需要继续改写文章."
#
#         st.session_state["platform"] = platform
#         st.session_state["llm_model"] = llm_model
#         st.session_state["temperature"] = temperature
#         st.session_state["prompt"] = prompt
#         st.session_state["article_generation_repeat_break_point"] = True
#         # langgraph 退出循环的判断条件
#         st.session_state["is_article_generation_complete"] = True
#
#         user_input = (f"模型: {llm_model}\n"
#                       f"温度: {temperature}\n"
#                       f"指令: {prompt}")
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         st.session_state.messages.append({
#             "role": "user",
#             "content": user_input,
#             "type": "text"  # 标识为文本类型
#         })
#         st.rerun()


async def create_graph(
        graph_class: Type[Graph],
        graph_input: Any,
        graph_config: dict,
        graph_llm: ChatOpenAI,
        graph_tools: list[BaseTool],
        graph_history_len: int,
):
    if st.session_state["checkpoint_type"] == "memory":
        if "memory" not in st.session_state:
            st.session_state["memory"] = get_checkpointer()
        checkpoint = st.session_state["memory"]
        graph_class = graph_class(llm=graph_llm,
                                  tools=graph_tools,
                                  history_len=graph_history_len,
                                  checkpoint=checkpoint)
        graph = graph_class.get_graph()
        if not graph:
            raise ValueError(f"Graph '{graph_class}' is not registered.")
        await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)
    elif st.session_state["checkpoint_type"] == "sqlite":
        checkpoint_class = get_checkpointer()
        async with checkpoint_class as checkpoint:
            graph_class = graph_class(llm=graph_llm,
                                      tools=graph_tools,
                                      history_len=graph_history_len,
                                      checkpoint=checkpoint)
            graph = graph_class.get_graph()
            if not graph:
                raise ValueError(f"Graph '{graph_class}' is not registered.")
            await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)
    elif st.session_state["checkpoint_type"] == "postgres":
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncConnectionPool(
            conninfo=Settings.basic_settings.POSTGRESQL_GRAPH_DATABASE_URI,
            max_size=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_MAX_SIZE,
            kwargs=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_KWARGS,
        ) as pool:
            checkpoint = AsyncPostgresSaver(pool)
            # NOTE: you need to call .setup() the first time you're using your checkpointer
            await checkpoint.setup()
            graph_class = graph_class(llm=graph_llm,
                                      tools=graph_tools,
                                      history_len=graph_history_len,
                                      checkpoint=checkpoint)
            graph = graph_class.get_graph()
            if not graph:
                raise ValueError(f"Graph '{graph_class}' is not registered.")
            await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)


async def update_state(graph: CompiledStateGraph, graph_config: Dict, update_message: Dict, as_node: str):
    # rich.print(update_message)  # debug

    # print("--State before update--")
    # # 使用异步函数来获取状态历史
    # state_history = []
    # async for state in graph.aget_state_history(graph_config):
    #     state_history.append(state)
    # rich.print(state_history)

    # 更新状态
    await graph.aupdate_state(config=graph_config,
                              values=update_message,
                              as_node=as_node)

    # print("--State after update--")
    # # 再次打印状态历史
    # state_history = []
    # async for state in graph.aget_state_history(graph_config):
    #     state_history.append(state)
    # rich.print(state_history)


async def graph_agent_page():
    # 初始化
    init_conversation_id()
    if "article_generation_init_break_point" not in st.session_state:
        st.session_state["article_generation_init_break_point"] = False
    if "article_generation_start_break_point" not in st.session_state:
        st.session_state["article_generation_start_break_point"] = False
    if "article_generation_repeat_break_point" not in st.session_state:
        st.session_state["article_generation_repeat_break_point"] = False
    if "is_article_generation_complete" not in st.session_state:
        st.session_state["is_article_generation_complete"] = False

    with st.sidebar:
        tabs_1 = st.tabs(["工具设置"])
        with tabs_1[0]:
            agent_graph_names = list_graph_titles_by_label(label="agent")
            selected_graph = st.selectbox(
                "选择工作流",
                agent_graph_names,
                format_func=lambda x: x,
                key="selected_graph",
                help="必选，不同的工作流的后端 agent 的逻辑不同，仅支持单选"
            )

            tools_list = list_tools()
            if selected_graph == "数据库查询机器人[Beta]":
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=["query_sql_data"],
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    default="query_sql_data",
                    help="仅可选择 SQL查询工具"
                )
            elif selected_graph == "自我反思机器人[Beta]":
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=list(tools_list),
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    default="search_internet",
                    help="支持多选"
                )
            else:
                # selected_tools demo: ['search_internet', 'search_youtube']
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=list(tools_list),
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    help="支持多选"
                )

            selected_tool_configs = {
                name: tool["config"]
                for name, tool in tools_list.items()
                if name in selected_tools
            }

        tabs_2 = st.tabs(["聊天设置"])
        with tabs_2[0]:
            history_len = st.number_input("历史对话轮数", 0, 20, key="history_len")

        st.tabs(["工作流流程图"])

    selected_tools_configs = list(selected_tool_configs)

    if selected_graph == "article_generation":
        st.title("自媒体文章生成")
        with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
            st.write("Hello 👋😊，我是自媒体文章生成 Agent，输入任意内容以启动工作流～")
    elif selected_graph == "数据库查询机器人[Beta]":
        st.title("数据库查询")
        with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
            st.write("Hello 👋😊，我是数据库查询机器人，输入你想查询的内容～")
    else:
        st.title("LLM 聊天")
        with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
            st.write("Hello 👋😊，我是聊天机器人，试着输入任何内容和我聊天呦～（ps: 可尝试选择多种工具）")
    # st.write(f"DEBUG: selected_graph = '{selected_graph}'")
    if selected_graph == "简历打分+面试模拟Agent":
        uploaded_file = st.file_uploader(
            "📎 上传简历（PDF/DOCX），或直接在下方输入文字",
            type=["pdf", "docx"],
            key="resume_uploader",
        )
    else:
        uploaded_file = None
    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            st.session_state["messages"] = []
            st.rerun()
        if selected_graph == "article_generation":
            user_input = cols[2].chat_input("尝试输入任何内容和我聊天呦 (换行:Shift+Enter)")
        elif selected_graph == "数据库查询机器人[Beta]":
            user_input = cols[2].chat_input("尝试输入任何内容和我聊天呦 (换行:Shift+Enter)")
        elif selected_graph == "简历打分+面试模拟Agent":
            user_input = cols[2].chat_input("或直接粘贴简历文本 / 输入对话内容 (换行:Shift+Enter)")
            if uploaded_file is not None and "last_uploaded_file" not in st.session_state:
                import os, tempfile
                st.session_state["last_uploaded_file"] = uploaded_file.name
                suffix = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    if suffix == ".pdf":
                        from chatchat.server.file_rag.document_loaders.mypdfloader import RapidOCRPDFLoader
                        loader = RapidOCRPDFLoader(tmp_path)
                    else:
                        from chatchat.server.file_rag.document_loaders.mydocloader import RapidOCRDocLoader
                        loader = RapidOCRDocLoader(tmp_path)
                    docs = loader.load()
                    resume_text = "\n".join(doc.page_content for doc in docs).strip()
                except Exception as e:
                    resume_text = f"[文件解析失败: {e}，请直接粘贴简历文本]"
                finally:
                    os.unlink(tmp_path)
                user_input = f"请对以下简历进行打分，然后开始面试模拟：\n\n{resume_text}"
                # st.write(f"已成功解析简历文本，{resume_text}")
            if uploaded_file is None:
                st.session_state.pop("last_uploaded_file", None)
        else:
            user_input = cols[2].chat_input("尝试输入任何内容和我聊天呦 (换行:Shift+Enter)")

    # get_tool() 是所有工具的名称和对象的 dict 的列表
    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]

    # 创建 llm 实例
    # todo: max_tokens 这里有问题, None 应该是不限制, 但是目前 llm 结果为 4096
    llm_model = st.session_state["llm_model"]
    llm = create_agent_models(configs=None,
                              model=llm_model,
                              max_tokens=None,
                              temperature=st.session_state["temperature"],
                              stream=st.session_state["streaming"])
    st.toast(f"已加载 LLM: {llm_model}")
    logger.info(f"Loaded llm: {llm}")

    # 创建 langgraph 实例
    graph_class = get_graph_class_by_label_and_title(label="agent", title=selected_graph)

    graph_instance = st.session_state["graph_dict"].get(selected_graph)
    if graph_instance is None:
        graph_png_image = get_img_base64(f"{selected_graph}.jpg")
        # if not graph_png_image:
        #     graph_png_image = graph.get_graph().draw_mermaid_png()
        #     logger.warning(f"The graph({selected_graph}) flowchart is not found in img, use graph.draw_mermaid_png() to get it.")
        st.session_state["graph_dict"][selected_graph] = {
            "graph_class": graph_class,
            "graph_image": graph_png_image,
        }
    st.toast(f"已加载工作流: {selected_graph}")

    # langgraph 配置文件
    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
    }
    logger.info(f"Loaded graph: '{selected_graph}', configurable: '{graph_config}'")

    st.sidebar.image(st.session_state["graph_dict"][selected_graph]["graph_image"], use_column_width=True)

    if selected_graph == "article_generation":
        # 初始化文章和图片信息
        if "article_links" not in st.session_state:
            st.session_state["article_links"] = ""
        if "image_links" not in st.session_state:
            st.session_state["image_links"] = ""
        if "article_links_list" not in st.session_state:
            st.session_state["article_links_list"] = []
        if "image_links_list" not in st.session_state:
            st.session_state["image_links_list"] = []

    # 前端存储历史消息(仅作为 st.rerun() 时的 UI 展示)
    # 临时列表，用于收集 assistant 的消息
    assistant_messages = []

    # 遍历 st.session_state.messages 并展示消息
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        is_last_message = message.get('is_last_message', False)

        if role == 'user':
            # 展示 user 消息
            with st.chat_message("user"):
                st.markdown(content)
        elif role == 'assistant':
            # 收集 assistant 消息
            assistant_messages.append(message)
            # 如果是最后一条 assistant 消息，立即展示
            if is_last_message:
                with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
                    for msg in assistant_messages:
                        if msg['is_last_message']:
                            st.markdown(msg['content'])
                        else:
                            with st.status(msg['node'], expanded=True) as status:
                                st.json(msg['content'], expanded=True)
                                status.update(
                                    label=msg['node'], state="complete", expanded=False
                                )
                # 清空临时列表
                assistant_messages = []

    # 对话主流程
    if user_input:
        st.session_state.messages.append(create_chat_message(
            role="user",
            content=user_input,
            node=None,
            expanded=None,
            type="text",
            is_last_message=True
        ))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the async function in a synchronous context
        graph_input = {"messages": [("user", user_input)]}
        await create_graph(graph_class=st.session_state["graph_dict"][selected_graph]["graph_class"],
                           graph_input=graph_input,
                           graph_config=graph_config,
                           graph_llm=llm,
                           graph_tools=tools,
                           graph_history_len=history_len)
        st.rerun()  # Clear stale containers
