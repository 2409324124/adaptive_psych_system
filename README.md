# CAT-Psych

一个可直接跑起来的自适应人格测评原型：

- FastAPI Web + API
- MIRT / CAT 自适应选题
- Big Five 50 题 starter bank
- 中文优先、英文辅助题干
- 可选大语言模型进行人格解析和简易分析

这个 README 只保留“怎么启动、怎么测、怎么改”三件事。

## 60 秒启动

### 1. 准备环境

```powershell
conda env create -f environment.yml
conda activate IPIP
```

如果环境已经存在：

```powershell
conda env update -n IPIP -f environment.yml --prune
```

### 2. 启动服务

```powershell
conda activate IPIP
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

## 最常用的 3 个模式

### 模式 A：本地直接跑

默认就是内存 session，适合开发和本地验证。

```powershell
conda activate IPIP
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

### 模式 B：开启本地 JSON 会话持久化

适合调试 session 生命周期、恢复本地测试数据。

```powershell
$env:CAT_PSYCH_SESSION_BACKEND="json"
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

可选参数：

- `CAT_PSYCH_SESSION_TTL_SECONDS`：session TTL，默认 `7200`
- `CAT_PSYCH_SESSION_DIR`：JSON session 文件目录，默认 `data/sessions/`

### 模式 C：接入 DeepSeek 生成大语言模型分析

不配置也能跑，只是会走 fallback 文案。

在项目根目录 `.env` 中放入：

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=...
DEEPSEEK_MODEL=...
```

然后正常启动：

```powershell
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

- `data/ipip_full_item_bank.json` 是资料档案，不是当前 Web/API 直接使用的在线题库。
- 当前默认交互路径仍围绕 50 题 starter bank。

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

### 跑全部测试

```powershell
conda run -n IPIP pytest -q
```

### 只跑 API 测试

```powershell
conda run -n IPIP python -m pytest tests\test_api.py -q
```

### 只跑 LLM / 翻译相关测试

```powershell
conda run -n IPIP python -m pytest tests\test_llm.py -q
```

## 常见开发命令

### 重建 IPIP 原始数据产物

```powershell
conda run -n IPIP python scripts\prepare_ipip_data.py
```

### 跑一个小型 adaptive session 仿真

```powershell
conda run -n IPIP python scripts\simulate_adaptive_sessions.py --max-items 12 --param-mode keyed
```

### 跑 stopping rule benchmark

```powershell
conda run -n IPIP python scripts\benchmark_stopping_rules.py --max-items 50 --param-mode keyed
```

### 生成 keyed mock 参数

```powershell
conda run -n IPIP python scripts\generate_key_aware_mock_params.py --output data\mock_params_keyed.pt
```

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
- DeepSeek 分析层是增强项，不是测量真值来源

## 伦理说明

本项目使用公开 IPIP 数据做原型验证。
不得抓取、复刻、逆向或替代受保护的 MMPI 题本、专有常模表或临床评分体系。
