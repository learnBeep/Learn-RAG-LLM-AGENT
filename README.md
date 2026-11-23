# 智能客服 Agent（扫地机器人领域）

基于 LangChain `create_agent` 构建的领域智能客服：模型按 ReAct 流程自主决定调用哪些工具、调用几轮，结合本地 RAG 知识库与用户使用记录，完成售前咨询、故障排查与个人使用报告生成。

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.5-1C3C3C)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)

---

## 功能特性

- **ReAct 智能体**：模型自行判断现有信息是否充分，自主编排 7 个工具的调用顺序，必要时发起多轮检索
- **RAG 知识库**：PDF / TXT 解析 → 递归分块 → 向量化 → Chroma 持久化；以文件 MD5 指纹做增量入库，避免重复加载污染检索结果
- **场景化提示词切换**：把「生成使用报告」建模为一次工具调用，中间件在工具执行后写入 `runtime.context` 标记，`dynamic_prompt` 在每次模型调用前据此切换主提示词与报告提示词，解决单一提示词下闲聊问答与结构化报告互相干扰的问题
- **工具调用可观测**：`wrap_tool_call` 中间件记录每次调用的工具名、入参与成功状态；`before_model` 记录每次模型调用的消息数与末条内容
- **并发保护**：检索摘要阶段用信号量限制模型并发调用，避免多工具并行触发时瞬时打满 API 配额
- **配置外置**：提示词、模型参数、向量库参数走 YAML；接口地址与密钥走环境变量，个人配置不进版本库

---

## 架构

```
┌──────────────┐   stream    ┌────────────────┐
│  Streamlit   │ ──────────► │   ReactAgent   │
│   (app.py)   │ ◄────────── │  (LangChain)   │
└──────────────┘             └───────┬────────┘
                                     │ ReAct 循环
                      ┌──────────────┴──────────────┐
                      ▼                             ▼
              ┌───────────────┐             ┌───────────────┐
              │ LLM (Qwen)    │             │  Tools × 7    │
              └───────────────┘             └───────┬───────┘
                                                    │
                    ┌───────────────────────────────┼──────────────────┐
                    ▼                               ▼                  ▼
            ┌───────────────┐             ┌───────────────┐    ┌──────────────┐
            │  RAG 检索     │             │ 外部使用记录  │    │ 天气 / 定位  │
            │ Chroma + 嵌入 │             │ records.csv   │    │   / 用户 ID  │
            └───────────────┘             └───────────────┘    └──────────────┘
```

一次典型请求：用户提问 → 主提示词 → 模型判断信息不足 → 调用 `rag_summarize` → 向量检索取回 top-k 片段 → 摘要提示词 + 参考资料 → 模型 → 若信息仍不足则再次调用工具 → 汇总输出。

---

## 技术栈

`Python 3.14` · `LangChain 1.5` · `LangGraph` · `Chroma` · `DashScope` · `Streamlit` · `PyYAML`

---

## 快速开始

### 1. 准备

Python 3.11 及以上，以及两把 API Key：

- 对话模型：任意 OpenAI 兼容接口（本项目使用阿里云百炼的 Qwen）
- 向量化模型：阿里云 DashScope 的 `text-embedding-v4`

### 2. 安装

```bash
git clone <仓库地址>
cd <仓库目录>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env`：

```dotenv
OPENAI_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
CHAT_MODEL_URL=https://<你的接口地址>/compatible-mode/v1
```

`utils/config_handler.py` 在启动时会调用 `load_dotenv()` 自动读取该文件，**不需要额外配置**。

> `CHAT_MODEL_URL` 可省略，省略时使用公开默认端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
> 环境变量读取是**在进程启动时完成一次**的：修改 `.env` 后需要重启 Streamlit（或重启 IDE 里已打开的终端）才会生效。

### 4. 构建知识库

首次运行前，把 `data/` 目录下的语料灌入向量库（会写入 `chroma_db/`）：

```bash
python -m rag.vetcor_store
```

> 必须用 `-m` 从项目根目录执行。直接 `python rag/vetcor_store.py` 会让 `sys.path` 变成 `rag/` 目录，导致 `ModuleNotFoundError: utils`。
> 已入库的文件会被 MD5 记录跳过，重复执行只加载新增内容。

### 5. 启动

```bash
streamlit run app.py
```

浏览器打开终端里给出的地址即可对话。

---

## 配置说明

### 环境变量

| 变量 | 必填 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 是 | 对话模型鉴权（`ChatOpenAI` 读取） |
| `DASHSCOPE_API_KEY` | 是 | 向量化模型鉴权（`DashScopeEmbeddings` 读取） |
| `CHAT_MODEL_URL` | 否 | 对话模型接口地址，缺省走公开端点 |

**加载优先级**：`load_dotenv()` 默认不覆盖已存在的变量，所以**操作系统环境变量优先于 `.env`**。同一台机器上两处都配置时，以环境变量为准；排查「改了 `.env` 不生效」时先检查这一点。

### 配置文件

| 文件 | 作用 |
| --- | --- |
| `config/rag.yml` | 对话模型与嵌入模型的名称、超时与重试 |
| `config/chroma.yml` | 向量库集合名、持久化路径、top-k、分块大小与重叠、允许的文件类型 |
| `config/agent.yml` | 外部使用记录 CSV 的路径 |
| `config/prompts.yml` | 三个提示词文件的位置 |
| `prompts/main_prompt.txt` | 智能体主提示词（工具清单与调用约束） |
| `prompts/rag_summarize.txt` | 检索摘要提示词 |
| `prompts/report_prompt.txt` | 报告生成场景提示词 |

---

## 目录结构

```
.
├── app.py                  # Streamlit 入口
├── agent/
│   ├── react_agent.py      # ReAct 智能体装配
│   └── tools/
│       ├── agent_tools.py  # 7 个工具的定义与实现
│       └── middleware.py   # 工具监控 / 模型调用钩子 / 动态提示词
├── rag/
│   ├── rag_service.py      # 检索 + 摘要链
│   └── vetcor_store.py     # 向量库封装、文档加载与增量入库
├── model/
│   └── factory.py          # 对话模型 / 嵌入模型的工厂
├── utils/                  # 配置加载、日志、路径、文件处理
├── config/                 # YAML 配置
├── prompts/                # 提示词模板
├── data/                   # 知识库语料与外部使用记录
└── logs/                   # 运行日志（运行时生成）
```

---

## 常见问题

**启动报 `Missing credentials`**
两把 Key 都没读到。检查 `.env` 是否在项目根目录、变量名是否拼对，以及当前终端是不是在改 `.env` **之前**就开着的（老终端不会自动读到新变量）。

**报 `ModuleNotFoundError: No module named 'utils'`**
用了 `python rag/vetcor_store.py`。改成 `python -m rag.vetcor_store`。

**改了提示词 / 配置没反应**
Streamlit 会缓存 `st.session_state` 里的 `ReactAgent`，改完配置需要刷新页面或重启进程。

**知识库里没有新加的文档**
建库脚本用 MD5 去重，确认新文件放在 `data/` 且后缀在 `config/chroma.yml` 的 `allow_knowledge_file_type` 里。

---

## 已知问题与后续计划

- [ ] 会话没有多轮记忆，每次提问相互独立
- [ ] 尚未服务化，缺少 FastAPI 接口与 SSE 流式输出
- [ ] 工具调用缺少超时、最大轮数与重复调用检测，模型可能反复用同样参数检索
- [ ] 天气、用户 ID、定位工具目前返回模拟数据
- [ ] 没有评测集，检索参数（chunk size、top-k）未做对比实验
- [ ] `model/factory.py` 在 import 时就实例化模型，环境变量缺失会导致导入即报错，后续应改为惰性初始化

---

## 说明

本项目为个人学习实践，在课程示例的基础上完成，并补充了配置分离、日志体系、增量入库与中间件监控等工程化处理。

`data/` 下的 PDF 与 DOCX 为第三方教学资料，如需完整知识库请自行放入后重新运行建库脚本；`data/external/records.csv` 中的用户与使用记录均为合成数据，不对应任何真实用户。
