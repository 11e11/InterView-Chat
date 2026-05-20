

[![Generic badge](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12-blue.svg)](https://pypi.org/project/pypiserver/)

📃 **InterView-Chat**

基于 Doubao-mini 等大语言模型与 LangGraph 等应用框架实现，开源、可离线部署的 RAG 与 Agent 应用项目。

---

## 概述

🤖️ 一种利用 [LangGraph](https://langchain-ai.github.io/langgraph/)
思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。

💡 针对想自主准备面试的候选人开发的简历评分及面试模拟出题Agent系统，用户可以自主管理知识库，建立私有个性化的知识库以实现个性化面试出题。

✅ 支持市面上主流的开源 LLM、 Embedding 模型与向量数据库，可实现全部使用**开源**模型**离线私有部署**。与此同时，本项目也支持
OpenAI GPT API 的调用，并将在后续持续扩充对各类模型及模型 API 的接入。

⛓️ 本项目实现原理如下图所示，过程包括加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 ->
在文本向量中匹配出与问句向量最相似的 `top k`个 -> 匹配出的文本作为上下文和问题一起添加到 `prompt`中 -> 提交给 `LLM`生成回答。


从文档处理角度来看，实现流程如下：

![实现原理图2](docs/img/langchain+chatglm2.png)



## 项目优势与定位

### 项目定位
一款面向`应届/社招面试候选者`的`轻量级`的囊括了`RAG`,`Agent`等场景的`LLM`简历评分+面试模拟出题应用`微服务`.


### 项目特色
- `Interview-Chat` 历史消息存在 `sqlite`(默认, 支持异步) 或 `PostgreSQL`(支持连接池+异步) 中, 方便用户统一管理;
- `Interview-Chat` 提供了 `graph` 注册器和 `tool` 注册器, 并提供了几种 `demo` 和规范供参考, 用户可以自主开发组装 `graph`, 并且也都是异步;
- `Interview-Chat` 对话的全部流程(`agent`和`rag`)均采用 `LangGraph` 来构建, 需要模型具备 `function call` 的能力.
- `Interview-Chat` 计划只保留一个对话接口, 如: `/chat`, 其他操作全部通过 UI. 
