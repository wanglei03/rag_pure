# RAG Agent 实战学习项目

面向"大模型应用算法研发"岗位准备的实操项目:从零实现一个企业知识库问答系统,
覆盖 **RAG 全链路 + Agent + MCP + 效果评估**。所有 stage 均可离线运行(mock 模型),
也可一键切换到 DeepSeek / 通义千问 / Ollama 真实模型。

## 快速开始

```bash
# 推荐 Python 环境:conda env rag (C:\Users\37482\.conda\envs\rag)
pip install -r requirements.txt

# 逐层学习(推荐顺序,每层 5 分钟)
python stages/stage01_parsing_chunking/demo.py
python stages/stage02_embedding_vectorstore/demo.py
python stages/stage03_retrieval/demo.py
python stages/stage04_generation/demo.py
python stages/stage05_agent/demo.py
python stages/stage06_mcp/demo.py
python stages/stage07_evaluation/demo.py
```

注意:GBK  locale 的 Windows 上 pip 安装含中文注释的 requirements 可能报
`UnicodeDecodeError: 'gbk' codec ...`,加 `PYTHONUTF8=1` 环境变量即可。

切换真实模型:复制 `.env.example` 为 `.env`,填入对应 Key 并修改 provider:

```bash
LLM_PROVIDER=deepseek          # 或 dashscope / ollama
EMBED_PROVIDER=dashscope       # DeepSeek 无 embedding 接口,需另配
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
```

## 项目结构

```
config/settings.py        # 全局配置:provider 切换、切分/检索参数
src/
  llm/                    # LLM 抽象:OpenAI 兼容 + 离线 mock
  embedding/              # Embedding 抽象:同上
  ingestion/              # 文档加载 + 中文感知切分 + 入库管线
  vectorstore/            # 向量库接口(numpy 实现,可换 Chroma/Milvus)
  retrieval/              # BM25 / 混合检索 RRF / 重排
  generation/             # 带引用约束的生成 + 引用回收
  agent/                  # Tool-Calling Agent(结构化决策循环)
  mcp/                    # MCP server(知识库标准化)+ client(线程桥接)
  eval/                   # 评测集 + 检索/生成指标 + 对比执行器
data/
  docs/                   # 示例知识库:5 份制度/规范/产品文档
  eval/questions.json     # 10 题评测集
stages/stage01..07/       # 逐层教程:demo.py + README(概念/设计/面试题)
```

## 学习路线与 JD 能力对照

| Stage | 内容 | 对应 JD 要求 |
|---|---|---|
| 01 | 文档解析与切分 | RAG 基础、知识库 |
| 02 | 向量化与向量存储 | RAG 基础、算法配套工具 |
| 03 | 混合检索 + 重排 | 搜索推荐、算法优化 |
| 04 | 带引用生成与拒答 | 产品体验 sense、幻觉抑制 |
| 05 | Agent 工具调用循环 | Agent、全链路研发 |
| 06 | MCP 服务化 | MCP |
| 07 | 指标体系与对比实验 | "通过算法优化技术指标提升产品体验" |

## 面试讲述建议(项目 → 岗位语言翻译)

- "mock/真实双模" → **模型网关与多供应商适配**的设计动机;
- "hybrid + RRF + rerank" → 检索优化如何**量化**(stage07 对比表就是证据);
- "引用回收 + 拒答" → 产品体验与**信任机制**;
- "结构化决策 Agent" → 与 LangChain 类框架的关系:原理自研,框架是语法糖;
- "MCP 线程桥接 + stdio 协议坑" → 真实工程踩坑记录,比背概念有说服力得多;
- "两套指标互证" → 对 LLM-as-judge 局限性的认知,超出"会调包"的层次。

## 已知边界(面试主动讲是加分项)

- 单轮问答,未做多轮指代消解与 query 改写(明确的扩展点);
- 向量库为 numpy 暴力检索,生产换 HNSW 索引(接口已预留);
- 表格/图片解析未覆盖(PDF 版式与 OCR 是生产 RAG 的最大坑之一);
- mock embedding 只有"共享词越多越相似"的弱语义,检索指标的**相对对比**有效,绝对值不代表真实模型水平。
