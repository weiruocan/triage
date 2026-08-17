# AGENTS.md — TRIAGE Framework

## 项目简介

TRIAGE（Three-level Routing and Intelligent Agent Guidance for Efficient Execution）是一个减少 LLM 智能体 Token 消耗的框架。核心思想：**让智能体像人一样积累经验，而非每次从零推理**。

## 项目结构

```
triage/                    # 仓库根目录
├── README.md              # 快速开始、两种接入方式
├── AGENTS.md              # 本文件，给 AI 读的行为准则
├── config.yaml            # LLM 配置（唯一配置来源）
├── setup.py               # pip install -e . 安装
│
├── triage/                # 核心 Python 包
│   ├── __init__.py        # 导出 triage_agent 装饰器
│   ├── model_api.py       # LLM API 封装
│   ├── agent.py           # 主智能体编排器
│   ├── decorator.py       # @triage_agent 装饰器
│   ├── cli.py             # CLI 入口（triage run/wrap）
│   ├── retrieval/         # 检索模块（编码器、数据库、编辑器）
│   ├── router/            # 路由模块（LLM 路由）
│   ├── agents/            # 子智能体
│   ├── skills/            # 技能模块（ExecutionFlow、ErrorAwareSQLSkill）
│   └── baseline/          # 基线系统（ReAct Engine、Orchestrator）
│
├── examples/              # 快速上手示例
│   ├── quickstart.py      # 5 分钟快速入门
│   └── security_monitor/  # 完整示例
│
├── data/                  # 数据（模型、轨迹库、数据库）
│   ├── models/            # 语义编码器
│   ├── trajectories/      # 轨迹库
│   └── security_monitor.db
│
└── experiments/           # 论文实验（用户可忽略）
```

## 核心概念

### 三级路由

- **第一级（直接复用）**：完全相同查询，0 Token 消耗
- **第二级（编辑复用）**：相似查询，编辑执行骨架，约 250 Token
- **第三级（完整 ReAct）**：全新查询，完整 ReAct 循环

### 冷启动

- 轨迹库为空时自动退化为纯 ReAct
- 积累 10+ 条轨迹后启用二级复用
- 积累 50+ 条轨迹后启用一级复用

## 使用方式

### 方式 A：装饰器模式（推荐）

```python
from triage import triage_agent

@triage_agent
def my_agent(query: str) -> str:
    """用户的原始智能体，无需改动"""
    # ... 用户已有的 ReAct 逻辑 ...
    return result
```

### 方式 B：CLI 包装模式

```bash
triage wrap --agent "python my_agent.py"
```

## 给 AI 的指令

当用户要求"帮我接入 TRIAGE"时：

1. 分析用户的工作流代码结构
2. 如果是一个函数 → 使用装饰器模式
3. 如果是一个脚本 → 使用 CLI 包装模式
4. 自动生成接入代码，不做多余改动
5. 引导用户验证：`pip install -e .` → 运行示例


## 语义编码器配置

TRIAGE 使用 sentence-transformers（all-MiniLM-L6-v2）做语义相似度检索。

**正确加载路径**（本地已下载，无需联网）：

```python
from sentence_transformers import SentenceTransformer
import os

# 正确路径 — 相对于项目根目录 v3/triage/
local_path = os.path.abspath("data/models/sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(local_path)

# 或在代码中设置离线模式避免联网重试
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
```

**路径说明**：
- 绝对路径：`meta-agent/data/models/sentence-transformers/all-MiniLM-L6-v2`
- 项目内路径：`v3/triage/data/models/sentence-transformers/all-MiniLM-L6-v2`（符号链接，指向物理位置）
- Git 状态：`data/models/` 已在 `.gitignore` 中，不会被提交
- 模型文件：18 个文件（含 model.safetensors、pytorch_model.bin、tokenizer.json 等）
- 嵌入维度：384 维
- 离线模式：必须设置 `TRANSFORMERS_OFFLINE=1` 和 `HF_HUB_OFFLINE=1`，否则会触发联网重试导致超时

**不要在代码中写 `SentenceTransformer("all-MiniLM-L6-v2")` 这种联网加载方式，会因 huggingface 无法连接而超时失败。**

## 模型调用规范

所有 LLM 调用配置从 `config.yaml` 读取，不硬编码。
配置加载优先级：**环境变量 > config.yaml**

### 环境变量方式（推荐，避免 Key 写入文件）

```bash
export TRIAGE_API_KEY="sk-xxx"
export TRIAGE_BASE_URL="https://api.openai.com/v1"
export TRIAGE_MODEL="gpt-4"
```

### 配置文件方式

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: ""          # 建议使用环境变量 TRIAGE_API_KEY
  model: "gpt-4"
  temperature: 0.5
  max_tokens: 8192
```

## 行为准则

- 核心包在 `triage/` 下，通过 `pip install -e .` 安装后可用 `import triage`
- 根目录只放 README、AGENTS、config、setup 等核心文件
- 所有新代码都放在 `triage/` 包内
- 对外暴露的 API 在 `triage/__init__.py` 中定义
- 不修改用户原有代码，只做包装


## 核心术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 轨迹 | Trajectory | 一条查询的完整执行记录 = 按序排列的节点列表 |
| 轨迹节点 | Node | 执行中的一步，包含 tool 调用或 LLM 生成 |
| 轨迹库 | Trajectory DB | 存储历史轨迹的数据库，支持语义检索 |
| 一级链路 | Level 1 / Direct Reuse | 相同查询直接返回缓存结果，0 Token |
| 二级链路 | Level 2 / Edit Reuse | 相似查询参考成功轨迹生成新内容，约 200 Token |
| 三级链路 | Level 3 / Full ReAct | 全新查询走完整 ReAct 循环，自动入库 |
| 经验参考生成 | Reference Generate | 二级链路的核心策略：参考成功轨迹的 SQL，让 LLM 在此基础上生成新 SQL |
| 冷启动 | Cold Start | 轨迹库为空时自动退化为纯 ReAct，使用中自动积累 |
| 语义编码器 | Semantic Encoder | sentence-transformers 模型，将查询转为 384 维向量用于相似度检索 |
| LLM 路由器 | LLM Router | 一次 LLM 调用，判断新查询走哪级链路 |
| 编辑策略 | Edit Strategy | 二级链路中对节点的处理方式：reference_generate（参考生成）或 no_change（不变） |
| 轨迹编辑器 | Experience Editor | 二级链路中负责编辑适配轨迹的模块 |
| 装饰器 | Decorator | `@triage_agent` 一行接入用户智能体的方式 |
| CLI 包装 | CLI Wrap | `triage wrap` 零代码改动接入的方式 |


## TaaS 核心理念

### 定义

**TaaS（Trajectory-as-a-Skill）** — 将历史执行轨迹（Trajectory）抽象为可复用的技能（Skill），实现"经验即服务"。

### 与业界 Skill 的本质区别

> Skill 和 TaaS 是同一个问题的两个极端：
> - **Skill（Function Calling）**：开发者预定义的确定性功能模块，固定输入输出 schema，泛化能力弱但执行确定
> - **TaaS（Trajectory-as-a-Skill）**：从执行中自动积累的轨迹，通过语义检索 + LLM 编辑适配新查询，泛化能力强但执行带概率性

任何时候讨论 Skill 与 TaaS，都要回到这个本质区别上来思考。

### TaaS 与 Skill 的关系

**TaaS 是 Skill 的超集，Skill 是 TaaS 的确定性子集。**

- **Skill**：开发者预定义的确定性功能模块，固定输入输出，执行 100% 确定
- **TaaS（轨迹即 Skill）**：从历史执行中自动积累的经验轨迹，可以像 Skill 一样被调用
  - 轨迹中的 `tool` 类型节点 = 确定性 Skill 调用（执行 100% 确定）
  - 轨迹中的 `llm_generate` 类型节点 = 概率性 Skill 调用（LLM 生成，带概率性）
  - 轨迹整体 = 一个"动态 Skill"——能通过语义检索被发现，通过 LLM 编辑适配变体



### 自动 Skill 提炼机制

TaaS 的终极目标是形成**正反馈循环**：

```
轨迹自动积累 → 高频模式自动提炼为细粒度 Skill → Skill 提供确定性执行 → 更多轨迹积累 → 更细的 Skill 提炼
```

**提炼规则：**
1. 同一条轨迹模式出现 N 次（默认 3 次）→ 自动提炼为一个 Skill
2. 提炼时分析轨迹中 LLM 生成的节点，提取可变参数（如 device_type、date）
3. 生成的 Skill 注册到 Skill 注册表，可供后续链路直接调用
4. 提炼后的 Skill 使用者优先使用（0 Token，100% 确定），而非 LLM 重新生成

**当前 Skill 粒度问题：**
- `execute_sql` 太粗——它只是"执行任意 SQL"，不知道具体查什么
- 需要的细粒度：`query_sensor_online_rate(device_type, date)`、`query_alarm_count(period, region)`
- 细粒度 Skill 从轨迹中自动提炼，而非开发者手动定义

**Skill 在 TaaS 中的角色：**
- **确定性锚点**：轨迹中可编辑的节点，优先用 Skill 调用（0 Token，确定）
- **编辑失败兜底**：LLM reference_generate 失败时，回退到 Skill 的原始调用
- **执行引擎**：所有 tool 类型节点本身就是 Skill 调用
- **自动提炼目标**：高频轨迹模式 → 永久 Skill

### 设计原则

- TaaS 是上层框架，Skill 是底层原子
- 一条轨迹 = 多个 Skill 节点 + 多个 LLM 节点的有序组合
- 轨迹中的 Skill 节点提供确定性，LLM 节点提供泛化能力
- 长期目标：高频轨迹模式自动提炼为永久 Skill，从"概率性经验"升级为"确定性能力"
