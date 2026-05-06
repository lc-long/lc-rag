# AGENTS.md

## 项目概览

Nova-RAG: 企业级 RAG 知识库问答系统，FastAPI + React + PostgreSQL/pgvector。

## 关键路径与端口

- 后端入口: `backend/src/api/server.py`，端口 **5000**
- 数据库: PostgreSQL **5433** (pgvector/pg16)
- 前端: `frontend/`，Vite dev server
- 测试脚本: `tests/test_novatech.py`（依赖后端运行在 `http://localhost:5000`）

## 启动命令

### PostgreSQL
```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=novarag -p 5433:5432 pgvector/pgvector:pg16
# 或
docker compose up -d
```

### 后端（使用 uv）
```bash
cd /home/lyy/Nova-RAG/backend
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 5000
```

### 前端
```bash
cd /home/lyy/Nova-RAG/frontend
npm install
npm run dev
```

## 环境变量

配置模板: `backend/.env.example` → 复制为 `backend/.env`

必填:
```
MINIMAX_API_KEY=
MINIMAX_GROUP_ID=
ALIYUN_API_KEY=
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/novarag
HF_ENDPOINT=https://hf-mirror.com
```

`server.py` 启动时通过 `load_dotenv()` 加载 `.env`，并强制设置 `HF_ENDPOINT`。

## 架构

- **混合检索**: pgvector 向量 + BM25 (jieba 分词) + RRF 融合 (k=40)
- **向量维度**: 1024 (Aliyun DashScope text-embedding-v3)
- **Parent-Child 分块**: Parent ~2000字符 / Child ~500字符
- **外部 API**: MiniMax M2.7 (LLM), Aliyun DashScope (Embedding/Reranker), Qwen-VL (OCR)

## RAG 调参

核心参数在 `backend/src/core/config.py`，可通过 `.env` 覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_PARENT_SIZE` | 2000 | 父chunk大小 |
| `CHUNK_CHILD_SIZE` | 500 | 子chunk大小 |
| `CHUNK_OVERLAP` | 100 | chunk重叠 |
| `RRF_K` | 40 | RRF融合常数 |
| `RERANK_MIN_SCORE` | 0.5 | Reranker最低阈值 |
| `RETRIEVER_TOP_K` | 8 | 最终返回chunk数 |
| `RECALL_MULTIPLIER` | 8 | 召回放大倍数 |
| `MAX_CONTEXT_TOKENS` | 6000 | 上下文上限 |
| `EMBED_MODEL` | text-embedding-v3 | 嵌入模型 |
| `RERANK_MODEL` | gte-rerank | 重排序模型 |
| `QUERY_PATTERNS_FILE` | - | Query改写JSON文件路径 |

## 测试

```bash
cd /home/lyy/Nova-RAG/tests && python test_novatech.py
```

## 代码检查

```bash
# 前端 lint
cd /home/lyy/Nova-RAG/frontend && npm run lint

# 后端（无内置lint，用uv run执行）
cd /home/lyy/Nova-RAG/backend && uv run python -c "import ast; ast.parse(open('src/api/server.py').read())"
```

## Git 提交规范

格式: `type(scope): description`，type 为 `feat/fix/docs/style/refactor/perf`，scope 为 `ui/api/rag/db/config`，**英文提交信息**，**勤提交**。

详见 `document/02_Engineering_Standards_and_Coding_Protocols.md`

## 技术栈

- Python 3.12+ / `uv` 管理
- React 19 + TypeScript + Tailwind CSS + Vite + Zustand
- FastAPI + Uvicorn
- PostgreSQL 16 + pgvector
