"""LLM 抽象层:统一 chat / chat_json 接口,支持任意 OpenAI 兼容端点 + 离线 mock。

面试点:为什么要抽象一层?——供应商可替换、可测试(离线 mock 让 CI/演示不依赖网络)、
上层业务零改动。这就是企业里说的"模型网关"的最小形态。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


def extract_json(text: str) -> dict:
    """从模型输出中稳健地提取第一个 JSON 对象(容忍 ```json 包裹和前后废话)。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"模型输出中没有 JSON 对象: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class BaseLLM(ABC):
    name: str = "base"

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        """messages 为 OpenAI 格式 [{"role": ..., "content": ...}, ...]"""

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """要求模型输出 JSON 并解析。默认用提示词约束 + 稳健解析,
        对不支持原生 json_mode 的端点(如部分 ollama 模型)也能工作。"""
        instructed = list(messages) + [
            {
                "role": "system",
                "content": "你的回复必须是一个合法 JSON 对象,不要输出任何 JSON 以外的内容。",
            }
        ]
        return extract_json(self.chat(instructed, temperature=temperature))


class OpenAICompatibleLLM(BaseLLM):
    """对接一切 OpenAI 兼容端点:DeepSeek / DashScope 兼容模式 / Ollama / vLLM / SiliconFlow。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI  # 延迟导入,让 mock 模式零依赖

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self._model = model
        self.name = f"openai-compatible:{model}"

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        kwargs: dict = {"model": self._model, "messages": messages, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        print(f"[OpenAICompatibleLLM][{self.name}] messages: {messages}")
        print(f"[OpenAICompatibleLLM][{self.name}] resp.choices[0].message.content: {resp.choices[0].message.content}")
        return resp.choices[0].message.content or ""


class MockLLM(BaseLLM):
    """离线 mock LLM:不联网、不花钱,让全链路(生成/Agent/评估)在无网环境也能演示。

    它用规则识别三类典型请求并给出结构化回复:
    1. RAG 问答(消息里带"参考资料")→ 基于资料模板化作答,演示引用链路;
    2. Agent 工具决策(消息里带"## 可用工具")→ 按规则返回 tool_call / final 的 JSON;
    3. 评估评委(消息里带"评委")→ 返回固定评分 JSON。
    换成真实模型后本类完全不参与业务代码路径。
    """

    name = "mock-llm"

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        text = "\n".join(str(m.get("content", "")) for m in messages)

        if "## 可用工具" in text:
            return self._decide_tool(text)
        if "评委" in text:
            return json.dumps({"score": 0.85, "reason": "mock 评委:离线模式下返回固定评分"}, ensure_ascii=False)
        if "【资料】" in text or "参考资料" in text:
            return self._answer_from_context(text)
        return (
            "我是离线 mock 模型(未配置真实 LLM)。"
            "在 .env 中设置 LLM_PROVIDER=deepseek/dashscope/ollama 并填入对应 API Key 后,"
            "这里将返回真实模型的回答。"
        )

    # ---- 规则 1:RAG 问答,只在资料区摘句作答;内容词对不上就拒答 ----
    @staticmethod
    def _answer_from_context(text: str) -> str:
        import jieba

        refuse = "根据现有资料无法回答这个问题。建议补充与问题相关的制度文档。"
        parts = text.split("问题:")
        ctx = parts[0]  # 只在资料区摘句,避免把问题本身拼进答案
        q = parts[-1].split("\n")[0].strip() if len(parts) > 1 else ""
        docs = re.findall(r"\[(\d+)\] 《(.+?)》", ctx)
        if not docs:
            return refuse

        # 强特征命中(问题中的专有 bigram/词在资料中出现)才算资料相关;全被停用 → 拒答
        generic = {"公司", "员工", "时间", "规定", "工作", "相关", "问题", "资料", "提供", "以上",
                   "进行", "支持", "属于", "可以", "需要", "申请", "怎么", "什么", "多少", "午餐",
                   "休息", "小时", "费用", "标准", "管理", "制度", "天内", "通过"}
        grams = {q[i : i + 2] for i in range(len(q) - 1)} | {t for t in jieba.lcut(q) if len(t) > 1}
        content_grams = {g for g in grams if len(g.strip()) > 1 and g not in generic}
        if not any(g in ctx for g in content_grams):
            return refuse

        sentences = re.findall(r"[^\n。;]{8,80}(?:。|;|$)", ctx)
        picked = [s.strip() for s in sentences if set(jieba.lcut(s)) & content_grams][:2]
        if not picked:
            return refuse
        return f"根据《{docs[0][1]}》的相关规定[1]:{' '.join(picked)} 以上信息摘自所提供的参考资料。"

    # ---- 规则 2:Agent 工具决策 ----
    @staticmethod
    def _decide_tool(text: str) -> str:
        has_kb_result = "【kb_search 结果】" in text
        has_calc_result = "【calculator 结果】" in text
        question_m = re.search(r"用户问题[:：]\s*(.+)", text)
        question = question_m.group(1).strip() if question_m else ""

        expr_m = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)*)", question)

        # 已拿到知识库结果,且问题需要计算 → 调计算器
        if has_kb_result and expr_m and not has_calc_result:
            return json.dumps({"tool": "calculator", "args": {"expression": expr_m.group(1)}}, ensure_ascii=False)
        # 已有知识库结果,不需要计算 → 总结收尾
        if has_kb_result:
            snippet = re.search(r"【kb_search 结果】\s*(.{10,120})", text, re.DOTALL)
            summary = snippet.group(1).replace("\n", " ") if snippet else "已完成检索"
            return json.dumps({"final": f"根据知识库检索结果:{summary}……(mock 总结)"}, ensure_ascii=False)
        # 没有知识库结果 → 先检索;含数字运算的问题在拿到结果后会补一次计算器调用
        if expr_m and not has_kb_result:
            # 先去重:mock 策略仍先查知识库,把数字相关描述一起带进去
            return json.dumps({"tool": "kb_search", "args": {"query": question}}, ensure_ascii=False)
        return json.dumps({"tool": "kb_search", "args": {"query": question}}, ensure_ascii=False)


def build_llm() -> BaseLLM:
    """工厂函数:按配置构造 LLM。业务代码只依赖 BaseLLM 接口。"""
    from config import settings

    cfg = settings.get_llm_config()
    if cfg["provider"] == "mock":
        print("[LLM] 使用离线 mock 模型")
        return MockLLM()
    if not cfg.get("api_key"):
        raise RuntimeError(
            f"provider={cfg['provider']} 需要 API Key,请在 .env 中设置对应 *_API_KEY,或改用 LLM_PROVIDER=mock"
        )
    print(f"[LLM] 使用 {cfg['provider']} 模型 {cfg['model']}")
    return OpenAICompatibleLLM(base_url=cfg["base_url"], api_key=cfg["api_key"], model=cfg["model"])
