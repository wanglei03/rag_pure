# Stage 02:向量化与向量存储

## 这一层解决什么

把 chunk 文本变成向量(embedding),存入向量库,让"找相关资料"变成"找最近邻向量"。
面试必考:余弦相似度 = 两个向量点积 / (模长之积);向量归一化后,**点积就是余弦**——
看 `NumpyVectorStore.search` 那一行 `self.vectors @ query_vec`,就这么简单。

## 代码位置

- `src/embedding/base.py`:`BaseEmbedder` 接口 + OpenAI 兼容实现 + `MockEmbedder`
- `src/vectorstore/store.py`:`NumpyVectorStore`(暴力检索 + npz 持久化 + 元数据过滤)
- `src/ingestion/pipeline.py`:`ingest_documents` / `build_kb`(入库主线,后面所有 stage 复用)

## 关键设计

1. **LLM 与 Embedding 分开配置**:`config/settings.py` 里 `LLM_PROVIDER` 和 `EMBED_PROVIDER` 是两个开关。
   为什么?DeepSeek 只有 chat 没有 embedding——生产上"生成用 A 家、向量化用 B 家"是常态。
2. **MockEmbedder 是确定性的**:同一句话任何时候编码结果相同(哈希投影),保证离线演示与测试可复现。
3. **向量库接口按生产设计**:`search(query_vec, top_k, where)` 这个签名在 Chroma/Milvus/pgvector 上长得一样,换库不改业务代码。

## 运行

```bash
python stages/stage02_embedding_vectorstore/demo.py
```

观察:前两句(都在讲年假)相似度显著高于第三句(数据库变更);`where={"doc_id": ...}` 过滤后只返回目标文档的结果。

## 面试高频问题

- **为什么向量归一化后点积=余弦?** 余弦 = 点积/(|a||b|),归一化后 |a|=|b|=1。归一化还方便用乘法累加,计算更快。
- **万级、百万级数据分别怎么存?** 万级暴力扫就够(本项目);十万级以上用 HNSW 近似索引(Chroma/Milvus/FAISS),O(N)→O(log N),牺牲少量精度换速度。
- **embedding 模型怎么选?** 中文场景 BGE-M3 / text-embedding-v3 是常见答案;选型看 MTEB 中文榜 + 自己业务的检索指标,不看宣传。
- **向量库里存什么元数据?** doc_id、heading、时间戳、权限标签等。多租户场景按权限过滤(where)必须在检索层做,不能靠生成层自律。
