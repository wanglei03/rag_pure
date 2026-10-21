"""Tool-Calling Agent:结构化决策循环(比自由文本 ReAct 更稳、更好评估)。

循环:决策(JSON:调哪个工具/给最终答案) → 执行 → 把观测结果回填 → 再决策,直到 final 或步数耗尽。

面试点:
- 与原生 function calling 的区别:这里是"prompt 工程版",任何模型(含 ollama 本地模型)都能跑;
  生产上优先用供应商原生 tool calling(更稳),但兜底方案和评估都绕不开这个思路。
- 为什么决策要求 JSON:可解析、可校验(对 args 按 JSON Schema 校验)、可回放评估。
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

from src.agent.tools import Tool
from src.llm.base import BaseLLM, build_llm, extract_json

AGENT_SYSTEM = """你是一个会使用工具的助手。每次回复必须是一个 JSON 对象,二选一:
1. 需要调用工具时:{{"tool": "工具名", "args": {{...}}}} —— args 必须严格符合该工具的参数 schema;
2. 已有足够信息回答用户时:{{"final": "给用户的最终答案"}}

## 可用工具
{tools}

决策原则:
- 信息不足时优先调工具,不要编造;
- 知识库类问题先用 kb_search;涉及数值计算时用 calculator;
- 最终答案使用简体中文。"""


@dataclass
class Step:
    index: int
    action: str  # "tool:kb_search" 或 "final"
    detail: str = ""       # 工具参数或最终答案
    observation: str = ""  # 工具执行结果


@dataclass
class AgentResult:
    question: str
    final_answer: str
    steps: list[Step] = field(default_factory=list)


class ToolCallingAgent:
    def __init__(self, llm: BaseLLM | None = None, tools: list[Tool] | None = None, max_steps: int = 6):
        self.llm = llm or build_llm()
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.max_steps = max_steps

    def _decision_messages(self, question: str, history: list[dict]) -> list[dict]:
        tools_desc = "\n".join(
            f"- {t.name}: {t.description} 参数schema: {json.dumps(t.parameters, ensure_ascii=False)}"
            for t in self.tools.values()
        )
        msgs: list[dict] = [{"role": "system", "content": AGENT_SYSTEM.format(tools=tools_desc)}]
        msgs.extend(history)
        msgs.append({"role": "user", "content": f"用户问题:{question}"})
        return msgs

    async def arun(self, question: str) -> AgentResult:
        history: list[dict] = []
        steps: list[Step] = []
        for i in range(1, self.max_steps + 1):
            decision = extract_json(self.llm.chat(self._decision_messages(question, history)))
            if "final" in decision:
                answer = str(decision["final"])
                steps.append(Step(index=i, action="final", detail=answer))
                return AgentResult(question=question, final_answer=answer, steps=steps)

            name = str(decision.get("tool", ""))
            args = decision.get("args", {}) or {}
            if name not in self.tools:
                obs = f"错误:不存在名为 {name!r} 的工具,可用:{list(self.tools)}"
                steps.append(Step(index=i, action=f"tool:{name}", detail=json.dumps(args, ensure_ascii=False), observation=obs))
            else:
                observation = self.tools[name].execute(args)
                if inspect.isawaitable(observation):
                    observation = await observation
                obs = str(observation)
                steps.append(Step(index=i, action=f"tool:{name}", detail=json.dumps(args, ensure_ascii=False), observation=obs))
            # 回填历史:模型能看到自己上一轮的决策和观测结果
            history.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            history.append({"role": "user", "content": f"【{name} 结果】\n{obs}"})
        return AgentResult(question=question, final_answer="达到最大步数仍未得出结论", steps=steps)

    def run(self, question: str) -> AgentResult:
        import asyncio

        return asyncio.run(self.arun(question))
