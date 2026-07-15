# Codex 订阅接入说明

本文说明 Multi-Agent Paper Reader 如何在完全本机、单用户部署中复用用户自己的 ChatGPT/Codex 订阅，以及模型路由、工具权限和安全边界。

## 1. 安装与登录

`requirements.txt` 固定了 OpenAI 官方 Codex 仓库的不可变源码归档及其 SHA-256，对应 Codex runtime `0.144.4`。这样可以在 PyPI 正式发布匹配版本前识别 GPT-5.6 目录中的 `max` 与 `ultra`，同时避免依赖用户机器上的全局 Codex CLI。

安装项目依赖并启动本机服务：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
npm --prefix frontend-prototype install
npm --prefix frontend-prototype run build
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

随后打开 `http://127.0.0.1:8000/`，进入 **Settings → Codex 订阅**：

1. 已有本机 Codex/ChatGPT 登录时，点击“刷新状态”即可复用。
2. 尚未登录时，可选择“浏览器登录”或“使用设备码”。
3. 登录成功后，在模型路由中主动选择 Codex、模型和推理档位并应用。

网页不接收 Codex token，也不读取或保存认证缓存内容；登录由官方 runtime 完成。断开连接会退出本机共享的 Codex 会话，可能使其他本机 Codex 客户端也需要重新登录。

## 2. 路由优先级

Codex 不会覆盖用户选择的其他厂商。`TEXT_PROVIDER` 是唯一的全局文本路由开关：

- 选择 `codex`：论文分析及默认追问走 Codex 订阅。
- 选择 `openai` 或其他厂商：走该厂商的 API 配置，不会因为本机已登录 Codex 而改道。
- 单篇论文追问仍可在输入框中为当次请求选择已配置的其他路由，不改写全局设置。

视觉模型继续与文本厂商绑定。选择 Codex 时，只有实时目录中声明支持图片输入的所选模型才能用于图表理解。

## 3. 实时模型与六档推理

账号实时 `model/list` 是 Codex 路由的唯一事实来源。后端只保留以下三个模型，并且只在账号真实返回时显示：

| 模型 | low | medium | high | xhigh | max | ultra |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.6 Sol | 可用 | 可用 | 可用 | 可用 | 可用 | 可用 |
| GPT-5.6 Terra | 可用 | 可用 | 可用 | 可用 | 可用 | 可用 |
| GPT-5.6 Luna | 可用 | 可用 | 可用 | 可用 | 可用 | 当前不可用 |

这是 2026-07-15 本机账号的实测目录。界面仍以每次实时返回为准：目录失败、为空、缺少某模型或缺少某档位时会明确禁用，而不是使用静态列表继续提交。

## 4. 每轮安全配置

所有 Codex 调用均使用：

- 临时线程：`ephemeral=true`；
- 不保存 Codex 历史：`history.persistence="none"`；
- 只读沙箱：`sandbox=read_only`；
- 拒绝审批：`approval_mode=deny_all`；
- 原生 Web Search：`web_search="live"`；
- 保留运行时原生的规划、工具发现、图像查看与图像生成能力；实际可用项由所选模型和固定版本 runtime 决定；
- 5.6 模型要求的 Code Mode `exec`/`wait` 继续存在；它们是无 Node、无直接文件系统和网络能力的 V8 工具编排层，不是 Shell；
- 不继承用户 Codex 配置中的 developer instructions、技能说明、项目文档、通知命令、Apps、Plugins、Hooks 和 Memories；
- 关闭 Shell、统一执行与任意文件写入；
- 启动前禁用并核验从用户 Codex 配置继承的 MCP Server，隔离失败即拒绝启动。

Web Search 结果在回答下方作为“外部资料”单独显示，不与论文原文证据混写。只接受 `http`/`https` 来源链接。图像生成仅在用户明确请求生成视觉内容时使用；普通论文分析不会主动消耗图像生成能力。它是普通文件写入禁令的唯一受控例外，生成物由官方 runtime 保存到本机 `CODEX_HOME/generated_images` 目录。

固定 runtime 没有用于隐藏 `apply_patch` 的有效配置开关，因此它可能仍出现在 Code Mode 的嵌套工具描述中；本项目的 `sandbox=read_only` 与 `approval_mode=deny_all` 会阻止实际落盘。Settings 将它标记为“可见但写入受阻”，而不是错误地声称工具不存在。

## 5. 论文专用工具边界

下面八项不是 Codex 全部原生工具的死白名单，而是宿主额外挂载的论文数据能力边界。当调用绑定到一篇当前论文时，宿主只挂载一个临时 `paper_reader` MCP Server。模型不能传入文件路径、历史 ID 或任意坐标，只能通过该 Server 调用以下只读能力：

| 工具 | 能力 | 主要限制 |
| --- | --- | --- |
| `paper_search_evidence` | 检索原文证据 | 最多 8 条，文本长度受限 |
| `paper_get_section` | 读取一个章节 | 标题或 1-based 编号，最多 16,000 字符 |
| `paper_get_page` | 读取 PDF 页文本 | 每次最多 2 页 |
| `paper_get_figure` | 读取图注与视觉摘要 | 只接受 `Fxxx` ID |
| `paper_get_table` | 读取表格 | 只接受 `Txxx`，最多 40 行 × 12 列 |
| `paper_get_visual_region` | 渲染图/表区域 | 只允许解析器验证过的 bbox，禁止整页回退 |
| `paper_recall_memory` | 召回长期记忆 | 仅当前论文 LangMem 命名空间 |
| `calculate` | 数值计算 | AST 白名单，不使用 `eval` |

每轮上下文写入权限为 `0600` 的随机 capability 文件，调用结束后删除；异常遗留文件会在后续创建上下文时按时限清理。MCP 工具均声明为只读、非破坏、幂等且不访问开放世界。

## 6. Ultra 子 Agent

只有 `ultra` 会启用 Codex multi-agent 功能：

- `max_threads=3`，即主 Agent 外最多两个子 Agent；
- `max_depth=1`，禁止递归扩张；
- 其他五档使用 `max_threads=1`，即使 5.6 模型元数据仍声明协作工具，也无法创建任何子线程；
- 子 Agent 继承相同只读沙箱、拒绝审批、Web Search、原生工具策略和论文专用工具边界；
- 其他五档同时关闭配置开关并加入“不调用协作工具”的线程指令，线程容量是最终硬约束。

调用轨迹只展示推理档位、是否使用 Web Search、论文工具名称/数量及子 Agent 数，不向前端暴露 token、认证文件、线程 ID 或本机路径。

## 7. 本机接口防护

登录、设备码和退出接口只接受：

- loopback 客户端地址；
- `localhost` 或 loopback IP 的 Host；
- 浏览器携带 Origin 时必须与请求 Host 同源。

这同时阻止远程调用、普通跨站请求和 DNS 重绑定。该设计不适用于公网或多人部署；公网部署必须使用厂商 API 凭据，并另行实现用户认证、配额和计费。

## 8. 兼容性与升级

后端会同时核对 SDK source revision、归档哈希、runtime Python 包版本、二进制版本及 `ReasoningEffort` 枚举。任一不匹配时，Settings 会显示兼容性错误并禁用 Codex 路由。

未来升级步骤：

1. 查阅官方 SDK、模型和认证文档；
2. 在隔离环境安装候选版本；
3. 验证 Sol/Terra/Luna 目录、六档枚举、登录流程及 MCP 配置字段；
4. 更新 requirements pin、源码 revision、归档 SHA-256 和 `_RUNTIME_VERSION`；
5. 运行完整测试、前端构建、真实账号最小调用及 MCP 进程泄漏检查。

官方资料：

- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk)
- [Codex models](https://learn.chatgpt.com/docs/models)
- [Codex authentication](https://learn.chatgpt.com/docs/auth)
- [Agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security)
- [MCP](https://learn.chatgpt.com/docs/extend/mcp)
- [Pinned OpenAI Codex source revision](https://github.com/openai/codex/commit/3f74f00295dcb1346340686bb09c5bfd4f0237c4)
