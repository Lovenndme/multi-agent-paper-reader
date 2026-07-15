# LangMem 记忆架构

## 结论

论文追问的长期记忆现已完全切换到 LangMem 0.0.30。旧版 Session Memory、主题文件、Side Query、Auto Dream 与自研更新协议不再参与运行时；LangMem 的 memory manager 是唯一负责“新增、更新、不保存、删除”决策的组件。

项目仍保留两层必要的产品集成：

- LangGraph `BaseStore` 的 SQLite 持久化适配器，使记忆沿用 `.paper-reader/history.sqlite3`，无需额外部署数据库。
- 本地 `paraphrase-multilingual-MiniLM-L12-v2` Embedding 索引，使回答前召回不消耗远程模型调用，并支持中英文论文问答。

这两层只负责存储和检索，不自行判断应该记住什么；记忆语义完全由 LangMem 管理。

## 数据流

```text
完整用户/助手回合
        │
        ├─ 回答完成后 ──> 后台 LangMem manager
        │                    ├─ insert
        │                    ├─ update
        │                    ├─ delete
        │                    └─ no-op
        │                          │
        │                    SQLiteBackedMemoryStore
        │                          │
        │                  history.sqlite3
        │
下一次用户问题
        │
        └─ 本地 MiniLM 向量检索 ──> 最多 3 条、低分返回空 ──> Prompt
```

## 记忆范围

每条记忆使用 `PaperReaderMemory` 结构，并按论文 ID 隔离：

- `user`：稳定的角色、专业背景、目标或职责。
- `feedback`：用户确认的回答偏好与纠错。
- `project`：无法从论文正文直接恢复的项目决策、目标、期限与复现上下文。
- `reference`：需要持续检查的稳定外部资料入口。

临时滚动位置、当前页面、可从论文重新解析的事实、凭证和密钥不应保存。用户明确要求忽略记忆时，本轮返回空长期记忆。

## 持久化与兼容

- 新表：`langmem_memories` 与 `langmem_migrations`。
- 命名空间：`("paper-reader", paper_history_id)`，不同论文不互相召回。
- 旧版 `chat_memories` 和文件记忆只导入一次；旧文件不被删除。
- 删除论文时对应 LangMem 记录由外键级联删除，并同步驱逐进程内索引。
- SQLite 原始对话继续用于历史展示，但不会作为长期记忆重新召回，确保“忘记”后旧消息不会把已删除信息带回 Prompt。

## 模型调用与失败语义

- 回答前：仅进行本地 Embedding 检索，不调用远程模型。
- 回答后：后台调用一次 LangMem manager；它复用本轮实际选择的厂商、模型与响应模式。
- 后台失败：不推进处理游标，下一轮可重试；论文回答本身不受阻塞。
- 普通回答的远程调用数由旧版通常约 4 次降为 2 次：正式回答 1 次、后台记忆管理 1 次。

## 主要实现位置

- `core/chat_memory.py`：会话 API、Prompt 组装和 LangMem 后台调度。
- `core/langmem_store.py`：结构化 schema、SQLite `BaseStore` 适配和兼容迁移。
- `core/semantic_search.py`：本地 Embedding 适配和缓存。
- `scripts/evaluate_memory.py`：真实模型行为、召回、重启、压力及回答 A/B 评测。

## 上游边界

LangMem 官方提供 memory manager 与 LangGraph store 集成。官方持久化文档列出的生产 Store 主要为 PostgreSQL、MongoDB 和 Redis，没有 SQLite Store；因此本项目的 SQLite adapter 是保持本地单机部署体验所需的薄适配层，而不是自研记忆决策框架。
