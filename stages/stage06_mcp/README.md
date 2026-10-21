# Stage 06:MCP(Model Context Protocol)

## 这一层解决什么

Stage 05 里工具是"本项目里的 Python 函数"——换个项目就得重写接入。
MCP 把能力(工具/资源/提示词)封装成**标准协议服务**,任何 MCP 客户端都能即插即用。
JD 里"熟悉 RAG、MCP、Agent"的 MCP,指的就是这个。

## 代码位置

- `../../src/mcp_tool`:FastMCP 服务,暴露 `kb_search` / `list_documents` / `calculator` 三个工具 + `kb://stats` 资源
- `../../src/mcp_tool`:stdio 客户端,后台线程跑事件循环,对 Agent 暴露为同步 Tool

## 关键设计

1. **stdio 传输**:server 以子进程方式启动,通过标准输入输出通信——不需要端口、天然适合本地工具。
   **经典坑①:stdio 模式下 stdout 是协议通道,任何 print 都会污染协议**——客户端逐行做 JSONRPC 解析,
   一句 `[build_embedder] ...` 就能让初始化直接失败。server 把加载期(入库/embedder/reranker)的
   所有 print 统一 `redirect_stdout(sys.stderr)`:协议走 stdout,日志走 stderr。
2. **经典坑② (Windows 实测踩雷)**:numpy/BLAS 这类带原生线程池的库, **不要在 MCP 事件循环的异步 handler 里首次 import**——
   OpenBLAS/MKL 的线程初始化与 anyio 工作线程会死锁,症状是 `tools/call` 收到请求后永不返回。
   解法见 `../../src/mcp_tool`:模块导入时就饥渴加载知识库,把重活全部挡在 `mcp.run()` 启动事件循环之前。
3. **客户端线程模型**:MCP 会话是异步的,Agent 循环是同步的。客户端独占一个后台线程跑事件循环,
   用 `run_coroutine_threadsafe` 桥接;退出清理必须在创建会话的同一个 task 里完成
   (anyio 的 cancel scope 绑定 task,跨 task 关闭会报 `Attempted to exit cancel scope in a different task`)。
4. **工具协议零魔改**:server 的工具 schema 直接来自 MCP 协议定义,Agent 的 Tool 只是薄封装。
   同一个 server 挂到 Claude Desktop、Qoder、自研系统,行为完全一致。

## 运行

```bash
python stages/stage06_mcp/demo.py        # stdio 传输
python stages/stage06_mcp/demo_http.py   # streamable-http 传输(同一套工具)
```

## 两种传输方式对比(面试必问)

| | stdio | streamable-http |
|---|---|---|
| 连接模型 | server 是客户端的子进程,管道即协议 | server 是独立网络服务,多客户端共享 |
| 适用场景 | 本地工具(知识库、文件系统、命令行) | 远程服务、多应用复用同一能力 |
| 生命周期 | 绑客户端进程,退出即销毁 | 独立运维,需要启停与探活 |
| 鉴权 | 天然隔离(本机进程) | headers 注入 token / 反代统一鉴权 |
| 就绪判断 | 无信号,只能约定或轮询协议 | **端口可连通 ≈ 服务就绪**(导入与加载完成后 uvicorn 才绑端口) |
| 会话 | 一对一 | `Mcp-Session-Id` 请求头维持会话,server 可设 `max_sessions` / idle 超时 |

工具定义与传输完全解耦:`build_server()` 一套工具,`run(transport=...)` 换传输。
真实模型下 Agent 走 HTTP 工具的执行轨迹(查知识库 → 计算器 → 汇总)与 stdio 版一致。

HTTP 传输的新增工程点(在 stdio 三坑之上):
4. **饥渴加载对 HTTP 同样 mandatory** —— uvicorn 的 handler 也在事件循环里跑,numpy 死锁依旧会发生;
5. 端口占用/防火墙弹窗;demo 用"先 bind(0) 选空闲端口"规避;
6. 无状态需求(`stateless_http=True`)与断线续传的取舍;鉴权层放 FastMCP(`token_verifier`)还是反向代理,是私有化交付的常见架构讨论。

## 面试高频问题

- **MCP 和 function calling 什么关系?** 互补层。function calling 是"模型输出结构化调用"的能力;
  MCP 是"工具以什么标准被暴露和发现"的协议。模型通过 MCP 发现工具,通过 function calling 发起调用。
- **MCP 和直接写 API 有什么区别?** 写 API 要每个客户端单独适配;MCP 一次封装处处可用,
  还带工具发现(list_tools)、资源读取、权限边界。类比:USB-C 之于充电线。
- **stdio 和 HTTP(SSE)传输怎么选?** 本地工具、无状态、单机 → stdio;远程服务、多客户端共享 → HTTP。
- **MCP 的安全边界?** server 侧定义工具白名单;敏感参数走环境变量注入;生产上再加鉴权和审计日志。
