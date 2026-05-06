# Nova-RAG

> 企业级 RAG 知识库问答系统 — 基于 FastAPI + React + PostgreSQL/pgvector

**后端端口**: 5000 | **数据库**: PostgreSQL 5433 (pgvector) | **前端**: Vite 5173

---

## 架构

```
┌──────────────┐      ┌─────────────────────────────────────┐
│   Frontend   │─────>│         Backend (Python/FastAPI)     │
│  React + TS  │ SSE  │  port 5000                          │
│  Tailwind    │<─────│  ├── API: docs, chat, conversations   │
│  Vite        │      │  ├── Core: chunker/embedder/retriever │
└──────────────┘      │  │           /llm/ocr                  │
                      │  └── Storage: PostgreSQL + pgvector   │
                      └──────────────┬──────────────────────┘
                                   │
                      ┌────────────┴────────────┐
                      │  PostgreSQL + pgvector  │
                      │  (port 5433, Docker)    │
                      └─────────────────────────┘
```

## 快速启动

### 1. 数据库

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=novarag -p 5433:5432 pgvector/pgvector:pg16
```

### 2. 后端

```bash
cd backend
cp .env.example .env
# 填入 MINIMAX_API_KEY, MINIMAX_GROUP_ID, ALIYUN_API_KEY

uv run uvicorn src.api.server:app --host 0.0.0.0 --port 5000
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev
```

## 环境变量 (.env)

| 变量 | 说明 |
|------|------|
| `MINIMAX_API_KEY` | MiniMax API Key (LLM) |
| `MINIMAX_GROUP_ID` | MiniMax Group ID |
| `ALIYUN_API_KEY` | 阿里云 DashScope API Key (Embedding/Reranker/OCR) |
| `DATABASE_URL` | PostgreSQL 连接串 (默认 `postgresql://postgres:postgres@localhost:5433/novarag`) |
| `HF_ENDPOINT` | HuggingFace 镜像 (默认 `https://hf-mirror.com`) |

## 核心流程

### 文档上传 → RAG 索引

```
上传 → PDF解析(pdfplumber+PyMuPDF) → OCR(Qwen-VL) → Parent-Child分块
     → Embedding(DashScope text-embedding-v3, 1024维) → pgvector向量库
     → BM25索引(jieba分词) → 完成
```

### 问答流程

```
用户问题 → Self-Query解析(doc_name/page_range) → QueryRewriter扩展
        → 向量检索(pgvector) + BM25检索 → RRF融合(k=30) → Reranker精排(gte-rerank)
        → Prompt压缩(关键词+句子评分) → LLM(MiniMax-M2.7 + DeepSeek fallback)
        → SSE流式输出(reasoning + answer双轨)
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/docs/upload` | 上传文档 (PDF/DOCX/XLSX/CSV/PPTX/MD/TXT) |
| GET | `/api/v1/docs` | 文档列表 |
| GET | `/api/v1/docs/{id}/preview` | 文档预览 (inline) |
| GET | `/api/v1/docs/{id}/download` | 文档下载 |
| GET | `/api/v1/docs/{id}/images` | 图片列表 |
| DELETE | `/api/v1/docs/{id}` | 删除文档 |
| POST | `/api/v1/docs/batch-delete` | 批量删除 |
| POST | `/api/v1/chat/completions` | 流式问答 (SSE) |
| GET | `/api/v1/conversations` | 会话列表 |
| GET | `/api/v1/conversations/{id}` | 会话详情 |
| DELETE | `/api/v1/conversations/{id}` | 删除会话 |

## 核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| **Chunker** | `ParentChildChunker` | Parent ~2000字符 / Child ~500字符，表格预拆分 + Markdown 标题分块 |
| **Embedder** | `AliyunEmbedder` | DashScope text-embedding-v3, 1024维，批量大小6 |
| **BM25** | `BM25Indexer` + jieba | 关键词检索，与向量互补 |
| **Retriever** | `HybridRetriever` | 向量+BM25 RRF融合 (k=30)，Self-Query元数据过滤 |
| **QueryRewriter** | `QueryRewriter` | LLM扩展 + 语义缓存(512条) + PRF fallback |
| **Reranker** | `AliyunReranker` | DashScope gte-rerank, 最低分0.5 |
| **Compressor** | `compress_chunks` | jieba关键词 + 句子评分，保留80% |
| **LLM** | `MinimaxClient` | MiniMax-M2.7 主调 + DeepSeek fallback，响应缓存(cosine>0.95) |
| **OCR** | `QwenVL` (DashScope) | PDF图片提取，最多15页，自适应页面选择 |

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_PARENT_SIZE` | 2000 | Parent chunk 大小 |
| `CHUNK_CHILD_SIZE` | 500 | Child chunk 大小 |
| `CHUNK_OVERLAP` | 100 | chunk 重叠 |
| `RECALL_MULTIPLIER` | 15 | 召回放大倍数 |
| `RETRIEVER_TOP_K` | 8 | 最终返回 chunk 数 |
| `RRF_K` | 30 | RRF 融合常数 |
| `RERANK_MIN_SCORE` | 0.5 | Reranker 最低阈值 |
| `MAX_CONTEXT_TOKENS` | 6000 | 上下文 token 上限 |
| `MAX_HISTORY_TOKENS` | 2000 | 对话历史 token 上限 |
| `EMBED_BATCH_SIZE` | 6 | Embedding 批量大小 |
| `OCR_MAX_PAGES` | 15 | OCR 最大页数 |

## 目录结构

```
Nova-RAG/
├── backend/
│   └── src/
│       ├── api/
│       │   ├── routes/          # docs, chat, conversations, citations
│       │   ├── server.py         # FastAPI 入口, port 5000
│       │   ├── models.py         # SQLAlchemy: Document, Conversation, MessageModel
│       │   ├── database.py       # PostgreSQL 连接
│       │   └── components.py     # 全局组件初始化
│       └── core/
│           ├── chunker/          # Parent-Child 分块, PDF/DOCX/XLSX/CSV/PPTX/MD 解析
│           ├── embedder/         # Aliyun DashScope Embedding
│           ├── llm/              # MiniMax-M2.7 + DeepSeek fallback
│           ├── ocr/              # Qwen-VL 图片理解
│           ├── retriever/        # Hybrid + Self-Query + QueryRewriter + Reranker
│           ├── storage/          # pgvector VectorStore
│           └── config.py         # 集中配置
├── frontend/                    # React 19 + TypeScript + Tailwind + Vite + Zustand
├── tests/                       # 测试脚本
└── uploads/                     # 上传文件 + OCR 图片存储
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 19 + TypeScript + Tailwind CSS + Vite + Zustand |
| 后端 | Python 3.12 + FastAPI + Uvicorn + SQLAlchemy |
| 数据库 | PostgreSQL 16 + pgvector |
| Embedding | Aliyun DashScope text-embedding-v3 (1024 维) |
| Reranker | Aliyun DashScope gte-rerank |
| LLM | MiniMax M2.7 + DeepSeek (fallback) |
| OCR | Qwen-VL (DashScope) |
| 分词 | jieba |
| BM25 | rank-bm25 |

## 测试

```bash
cd tests && python test_novatech.py
```

## 代码检查

```bash
# 后端语法检查
cd backend && uv run python -c "import ast; ast.parse(open('src/api/server.py').read())"

# 前端 lint
cd frontend && npm run lint
```