# Stage 01:文档解析与切分

## 这一层解决什么

RAG 的第一道工序:把 PDF/Word/Markdown 文档变成"适合检索的短文本块(chunk)"。
切分质量直接决定检索上限——chunk 太大,向量被平均化,检索不精确;chunk 太小,
关键信息失去上下文(比如"住宿上限 600 元"切走了"北京、上海"这个前提)。

## 代码位置

- `src/ingestion/loaders.py`:按扩展名分发加载(md/txt/pdf/docx),pdf/docx 为可选依赖
- `src/ingestion/chunker.py`:`split_paragraphs`(先按段落)→ `recursive_chunk`(贪心装箱 + overlap 回退)→ `chunk_document`(记录归属标题)

## 关键设计

1. **先段落、后字数**:中文文档天然以段落为语义单元,硬按 200 字切会把一段话拦腰截断;
2. **overlap 回退**:上一块尾部带 `chunk_size/5` 左右的字符进下一块,边界句子不至于断义;
3. **元数据即线索**:每个 chunk 记录 `doc_id / heading / chunk_index`,检索结果可以直接展示"出自哪份文档哪一节",评估时也能按文档粒度判命中。

## 运行

```bash
python stages/stage01_parsing_chunking/demo.py
```

观察:chunk_size 从 160 → 450,块数下降、单块变长;打印出的每块都带归属标题。

## 面试高频问题

- **chunk size 怎么定?** 没有银弹,看文档类型和 embedding 模型上下文(中文 200–400 字常见)。
  工程做法:准备 20~50 个典型问题,用不同 chunk 参数跑 hit rate,选指标最好的——这就是"用算法优化技术指标提升产品体验"的落地姿势。
- **语义切分和固定切分怎么选?** 固定切分便宜稳定适合结构化文档;语义切分(按 embedding 相似度断句)适合无结构长文,但成本高、边界抖动。生产上常见组合:先标题/版面切,再对超长块二次固定切。
- **表格、图片怎么处理?** 这是 RAG 落地最大的坑之一。表格要还原成行/结构化文本(或 HTML),图片走 OCR/多模态模型;本项目为聚焦链路暂不展开,面试主动提这一点是加分项。
