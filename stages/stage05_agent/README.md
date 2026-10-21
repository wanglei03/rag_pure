# Stage 05:Agent(工具调用循环)

## 这一层解决什么

Stage 01–04 是"一问一答"的固定链路;Agent 把**决策权交给模型**:
查知识库、做计算、看时间,由模型根据问题自己编排,而不是工程师写死 if-else。
这正是 JD 里"具备全链路的 AI 研发能力"的核心构件。

## 代码位置

- `src/agent/tools.py`:Tool 数据类(kb_search / calculator / list_documents / current_time)+ 基于 ast 的安全计算器
- `src/agent/agent.py`:`ToolCallingAgent` —— 结构化决策循环

## 关键设计

1. **结构化决策而非自由文本 ReAct**:每轮要求模型只输出 JSON——`{"tool": ..., "args": ...}` 或 `{"final": ...}`。
   好处:可解析、可校验、可回放评估;任何模型(含本地 ollama)都能跑,不依赖原生 function calling。
2. **观测结果回填**:工具执行结果以 `【工具名 结果】` 形式追加进对话历史,模型据此决定下一步——这就是 Agent 的"记忆"。
3. **步数上限 + 错误反馈**:max_steps 防死循环;调错工具名时把错误信息喂回去让模型自我纠正,而不是直接抛异常。
4. **工具的错误也是信息**:`calculator` 遇到非法表达式返回错误字符串而非抛异常——模型看到错误可以换表达式重试。

## 运行

```bash
python stages/stage05_agent/demo.py
```

mock 模式下模型按规则走"查知识库 → 调计算器 → 总结"三步;换真实模型后轨迹由模型自主决策。

## 面试高频问题

- **和 LangChain 的 Agent 什么区别?** 这里是最小实现,思想相同:决策-执行-观测循环。讲清原理后,框架只是语法糖;而且知道原理才能调优(比如轨迹为什么跑偏、该加什么约束)。
- **模型一直调工具不收敛怎么办?** max_steps 兜底 + 在 prompt 里明确收尾条件;评估时统计步数分布,步数异常的问题单独分析。
- **工具调用的安全怎么保证?** 三层:参数按 JSON Schema 校验;执行层最小权限(计算器禁 eval 用 ast 白名单);危险操作(写库、转账)加 human-in-the-loop 确认节点。
- **什么时候该用 Agent 而不是固定 pipeline?** 步骤可预知、分支少 → 固定 pipeline 更稳更便宜;步骤需动态决策、工具种类多 → Agent。生产上很多"Agent"其实是"LLM 路由 + 固定 pipeline",成本可控。
