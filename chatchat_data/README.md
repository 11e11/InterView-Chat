# 知识库使用说明与Chunk设计说明

## 文件清单

| 文件名 | 主题 | 估算Token | 建议Chunk数 |
|--------|------|-----------|-------------|
| Agent_Engineering_Core.md | Agent架构/Prompt/记忆/工具调用 | ~4570 | 12个 |
| Advanced_RAG_SOP.md | RAG调优/混合检索/Rerank/上下文压缩 | ~4685 | 12个 |
| Backend_Scalability.md | 流式传输/任务队列/限流熔断 | ~5820 | 15个 |

---

## Chunk切块建议（针对BGE-large-zh-v1.5的512 Token上限）

### 切块策略
三个文件均按以下策略设计：

1. **以二级标题（##）为自然切块边界**
   每个 `##` 章节约 200-500 Token，天然适合 400 Token 的 Chunk 上限

2. **每个Chunk开头含关键词标签**
   格式：`知识名-子主题-具体概念`，确保小切块单独存在时仍可被精准召回
   示例：`RAG多路召回-BM25关键词检索特点` 这行文字确保即使只检索到这个Chunk，也能明确知道它属于哪个知识域

3. **代码块不跨Chunk切割**
   所有代码块均设计在单个 Chunk 内（最长代码块约 350 Token），避免代码被截断失去语义

### Chatchat上传设置建议

```
Chunk Size：400 Token
Chunk Overlap：50 Token（对叙述性段落）/ 0 Token（对代码块和QA对）
分隔符优先级：\n## > \n### > \n\n > \n
```

---

## RAG索引标签说明

每个文件顶部的 `<!-- RAG索引 -->` 注释是面向人类开发者的说明，上传到Chatchat后会被包含在第一个Chunk中，帮助向量模型理解文件整体主题。

每个 `##` 章节开头的粗体标签（如 `RAG多路召回-BM25关键词检索特点：`）是面向向量检索的关键设计：
- 即使该Chunk被切割出来单独存在，这个前缀也能让BGE正确理解其语义
- 面试官提问"BM25是什么"时，这个前缀确保该Chunk能被召回

---

## 扩展建议

后续可补充的知识文件：
- `LLM_Evaluation_Metrics.md`：大模型评测指标（BLEU/ROUGE/BERTScore/GPT-4 Judge）
- `Vector_DB_Comparison.md`：向量数据库选型（FAISS/Milvus/Chroma/Weaviate对比）
- `Prompt_Security.md`：Prompt注入攻击与防御（Jailbreak/Indirect Injection）
- `LLM_Finetuning_SOP.md`：LoRA/QLoRA微调流程与数据准备
