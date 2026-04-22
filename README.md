# CAT-Psych

一个可直接跑起来的自适应人格测评原型：

- FastAPI Web + API
- MIRT / CAT 自适应选题
- Big Five 50 题 starter bank
- 中文优先、英文辅助题干
- 可选大语言模型进行人格解析和简易分析
- 默认 CPU 运行，方便 Linux 开发和服务器部署

这个 README 只保留“怎么启动、怎么测、怎么部署”三件事。

## 60 秒启动

### 方式 A：Linux / pip CPU 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

打开：

```text
http://127.0.0.1:8000
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 方式 B：Conda CPU 环境

```powershell
conda env create -f environment.yml
conda activate IPIP
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

如果环境已经存在：

```powershell
conda env update -n IPIP -f environment.yml --prune
```

### 方式 C：Docker Compose

```bash
docker compose up -d --build
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

只构建镜像：

```bash
docker build -t cat-psych:cpu .
```

## Docker 部署说明

默认 Compose 服务会：

- 使用 `python:3.11-slim`
- 安装 CPU 版 `torch==2.5.1`
- 暴露 `8000:8000`
- 开启 JSON session 持久化
- 将宿主机 `./data` 挂载到容器 `/app/data`

这意味着：

- `data/ipip_items.json`、`data/ipip_items_zh.json`、`data/mock_params_keyed.pt` 是运行必需资产
- `data/cat_psych.db` 会作为 SQLite 结果库保留在宿主机
- `data/sessions/` 会保存 JSON session
- 服务器迁移或备份时，至少要保留 `data/`

可选大语言模型配置放在 `.env` 中，参考 `.env.example`：

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=
DEEPSEEK_MODEL=
```

不配置也能运行，会使用 fallback 分析文本。

## 常用运行模式

### 本地内存 session

适合开发和快速验证。

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

### 本地 JSON session 持久化

适合调试 session 生命周期和恢复本地测试数据。

```bash
export CAT_PSYCH_SESSION_BACKEND=json
export CAT_PSYCH_SESSION_DIR=data/sessions
export CAT_PSYCH_SESSION_TTL_SECONDS=7200
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Windows PowerShell：

```powershell
$env:CAT_PSYCH_SESSION_BACKEND="json"
$env:CAT_PSYCH_SESSION_DIR="data/sessions"
$env:CAT_PSYCH_SESSION_TTL_SECONDS="7200"
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

## 你现在实际会得到什么

### Web 侧

- 进入首页后可直接创建 session
- 题目按 CAT 路由逐题发放
- 中文题干优先显示，英文原文作为辅助
- 支持 early stop、confirmation window、stability gate
- 完成后可生成结果报告页与分享链接

### API 侧

常用接口：

- `POST /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/next`
- `POST /sessions/{session_id}/responses`
- `POST /sessions/{session_id}/comments`
- `GET /sessions/{session_id}/result`
- `GET /sessions/{session_id}/export`
- `GET /results/{session_id}`

## 数据现状

当前在线题库不是 3320 题全量库，而是 50 题 starter bank。

- 当前自适应题库：`data/ipip_items.json`
- 当前中文翻译表：`data/ipip_items_zh.json`
- 当前题目分配表：`data/ipip_item_assignment_table.json`
- 当前默认参数：`data/mock_params_keyed.pt`
- 历史/实验参数：`data/mock_params.pt`
- 全量抓取档案：`data/ipip_full_item_bank.json`

补充说明：

- `data/ipip_full_item_bank.json` 是资料档案，不是当前 Web/API 直接使用的在线题库
- 当前默认交互路径仍围绕 50 题 starter bank
- CUDA 不再是默认开发/部署路径，如需 GPU 实验请单独准备 CUDA 环境

## 核心配置

当前默认交互配置大意如下：

- `param_mode=keyed`
- `max_items=30`
- `min_items=5`
- `coverage_min_per_dimension=2`
- `stop_mean_standard_error=0.65`
- `stop_stability_score=0.7`

这些参数可以通过 Web 面板或创建 session 的请求体调整。

## 测试

### 确认 CPU PyTorch

```bash
python -c "import torch; print(torch.__version__); assert not torch.cuda.is_available()"
```

### 跑全部测试

```bash
python -m pytest -q
```

### 跑主要回归测试

```bash
python -m pytest tests/test_api.py tests/test_irt.py tests/test_llm.py -q
```

### 在 Docker 镜像中测试

```bash
docker build -t cat-psych:cpu .
docker run --rm cat-psych:cpu python -c "import torch; print(torch.__version__); assert not torch.cuda.is_available()"
docker run --rm cat-psych:cpu python -m pytest tests/test_api.py tests/test_irt.py tests/test_llm.py -q
```

## 常见开发命令

### 重建 IPIP 原始数据产物

```bash
python scripts/prepare_ipip_data.py
```

### 跑一个小型 adaptive session 仿真

```bash
python scripts/simulate_adaptive_sessions.py --max-items 12 --param-mode keyed
```

### 跑 stopping rule benchmark

```bash
python scripts/benchmark_stopping_rules.py --max-items 50 --param-mode keyed
```

### 生成 keyed mock 参数

```bash
python scripts/generate_key_aware_mock_params.py --output data/mock_params_keyed.pt
```

## 服务器还需要补什么

- 安装 Docker Engine 和 Docker Compose plugin
- 开放或反向代理 `8000` 端口
- 如果公网访问，建议补 Nginx/Caddy、HTTPS、域名和基础限流
- 给 `data/cat_psych.db` 和 `data/sessions/` 做备份策略
- DeepSeek 等密钥只放服务器环境变量或 `.env`，不要提交真实密钥
- 如果未来多服务器部署，需要把 SQLite/session 存储迁移到外部数据库或共享存储
- 建议后续补 GitHub Actions，在 Linux 上跑 CPU pytest 和 Docker build

## 项目入口

如果你要继续开发，先看这几个文件就够了：

- `api/app.py`：Web/API 入口，session/result 生命周期
- `services/assessment_session.py`：CAT 会话状态机、early stop、confirmation
- `engine/irt_model.py`：默认参数与题库入口
- `web/app.js`：前端交互与结果页渲染
- `tests/test_api.py`：主回归测试

## 现阶段边界

- 当前是 Big Five 自适应测评原型，不是临床诊断工具
- 当前线上题库仍是 50 题 starter bank，不是 3320 题全量自适应版本
- 大语言模型分析层是增强项，不是测量真值来源

## 伦理说明

本项目使用公开 IPIP 数据做原型验证。
不得抓取、复刻、逆向或替代受保护的 MMPI 题本、专有常模表或临床评分体系。
