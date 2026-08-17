# TRIAGE

**Three-level Routing and Intelligent Agent Guidance for Efficient Execution**

通过复用历史经验轨迹减少 LLM 智能体 Token 消耗。核心思想：**让智能体像人一样积累经验，而非每次从零推理**。

## 效果

| 指标 | 基线 ReAct | TRIAGE | 节省 |
|------|-----------|--------|------|
| 总 Token（237 条查询） | 127,095 | 57,007 | **55.1%** |
| 直接复用（L1，0 Token） | — | 38.8% 的查询 | 92/237 |
| 编辑复用（L2，部分 Token） | — | 49.8% 的查询 | 118/237 |
| 跨领域平均 | — | 59.5% | 3 个领域 (ToolBench) |

## 环境要求

- **Python**: 3.10+
- **LLM API**: 任何兼容 OpenAI 格式的 API（deepseek、OpenAI、vLLM 等）
- **可选**（语义相似度检索需要）：`sentence-transformers`

## 安装

```bash
# 方式 A：从源码安装（目前推荐）
git clone https://github.com/weiruocan/triage.git
cd triage
pip install -e .

# 方式 B：从 PyPI 安装（即将上线）
pip install triage-agent
```

## 快速开始（5 分钟）

### 第 1 步：配置 LLM API

```bash
export TRIAGE_API_KEY="sk-xxx"
export TRIAGE_BASE_URL="https://api.openai.com/v1"
export TRIAGE_MODEL="gpt-4"
```

### 第 2 步：运行内置示例

```bash
# 模拟演示（无需真实 LLM，展示三级路由机制）
python examples/quickstart.py

# 真实 LLM 端到端演示（冷启动 → 轨迹积累 → 复用）
python examples/end_to_end.py
```

### 第 3 步：接入你自己的项目

**方式 A：装饰器模式（推荐，一行代码）**

```python
from triage import triage_agent

@triage_agent
def my_agent(query: str) -> str:
    """你的原始智能体，无需任何改动"""
    # 你的 ReAct 逻辑...
    return result

# 使用方式完全不变——TRIAGE 自动处理缓存
print(my_agent("What is the online rate of sensors?"))
```

**方式 B：CLI 包装模式（零代码改动）**

```bash
# 通过 TRIAGE 运行单条查询
triage run --query "What is the online rate of sensors?"

# 包装你已有的 agent 脚本
triage wrap --agent "python my_agent.py"

# 生成对比报告
triage report
```

**方式 C：手动编排（灵活控制）**

```python
from triage import TriageOrchestrator

orchestrator = TriageOrchestrator()
result = orchestrator.run(
    query="What is the online rate of sensors?",
    fallback=my_agent,
)
```

### 第 4 步：验证效果

```bash
# 查看轨迹库状态
triage report

# 你会看到：
# - 冷启动：所有查询走完整 ReAct（自动存储轨迹）
# - 10+ 条后：二级编辑复用启用
# - 50+ 条后：一级直接复用启用
# - Token 节省实时展示
```

## 工作原理

```
用户查询
    │
    ▼
┌─────────────┐     语义编码器 (all-MiniLM-L6-v2)
│  大模型路由 │──→  余弦相似度检索
└──────┬──────┘     与轨迹库对比
       │
    ┌──┴──┐
    │     │
    ▼     ▼
  L1:     L2:             L3:
  直接复用  编辑复用         完整 ReAct
  0 Token  ~200 Token      (新查询，
  (精确匹配  (参考经验         自动入库)
  直接返回)   + 修改生成)

88.6% 的查询受益于复用 → 55.1% Token 节省
```

### 三级路由

| 层级 | 触发条件 | Token 消耗 | 查询占比 |
|------|---------|-----------|---------|
| **L1 — 直接复用** | 余弦相似度 ≥ 0.90 | **0** | 38.8% |
| **L2 — 编辑复用** | 余弦相似度 ≥ 0.65 | **~200** | 49.8% |
| **L3 — 完整 ReAct** | 余弦相似度 < 0.65 | 全量 | 11.4% |

### 冷启动 → 预热 → 成熟

| 阶段 | 轨迹数量 | 行为 |
|------|---------|------|
| ❄️ 冷启动 | 0 | 所有查询走完整 ReAct，自动存储轨迹 |
| 🔥 预热 | 10+ | 启用二级编辑复用 |
| 🏆 成熟 | 50+ | 启用一级直接复用 |

**无需手动配置。TRIAGE 在使用中自动学习——使用越多，效率越高。**

## 核心概念

### TaaS：Trajectory-as-a-Skill（轨迹即技能）

TRIAGE 的理论核心：将历史执行轨迹抽象为可复用的技能。

| 维度 | Skill（函数调用） | TaaS（轨迹即技能） |
|------|-----------------|-------------------|
| 定义方式 | 开发者手动定义 | 从执行中自动积累 |
| 粒度 | 粗粒度（整个函数） | 节点级，可编辑 |
| 泛化能力 | 弱（固定 schema） | 强（语义检索 + LLM 编辑） |
| 冷启动 | 需预先写好所有 Skill | 自动从 0 积累 |
| 维护成本 | 高（新功能 = 新 Skill） | 低（自动入库、自动衰减） |
| 确定性 | 强 | 概率性（LLM 编辑） |

### 自动 Skill 提炼

高频轨迹模式（≥ 3 次出现）自动提炼为确定性 Skill：

```
轨迹积累 → 模式分析 → 参数提取 → Skill 注册 → 0 Token 确定性执行
```

形成正反馈循环：**使用越多，效率越高**。

## 项目结构

```
triage/
├── README.md                 # 英文文档
├── README.zh.md              # 中文文档（本文件）
├── AGENTS.md                 # AI 行为准则（供 Codex 使用）
├── triage/                   # 核心 Python 包
│   ├── model_api.py          # LLM API 封装
│   ├── agent.py              # 智能体编排器
│   ├── decorator.py          # @triage_agent 装饰器
│   ├── cli.py                # CLI 入口
│   ├── retrieval/            # 检索模块（编码器、数据库、编辑器）
│   ├── router/               # 大模型路由
│   ├── skills/               # 技能模块（执行流、Skill 提炼）
│   └── baseline/             # 基线系统
├── examples/                 # 示例项目
├── experiments/              # 实验脚本与数据
├── docs/                     # 论文与实验报告
└── tests/                    # 单元测试
```

## 论文

完整论文（中文）见 `docs/paper/`，实验数据见 `docs/experiments/`。英文版将提交至 arXiv。

```bibtex
@misc{triage2026,
  title={TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution},
  author={TRIAGE Contributors},
  year={2026},
  publisher={GitHub},
  howpublished={\\url{https://github.com/weiruocan/triage}}
}
```

## AI 辅助声明

本项目的代码和文档使用了 AI 辅助工具（Codex CLI）进行开发。
所有核心设计、算法选择、实验方案和数据分析均由人类作者经过反复设计、迭代纠错和调整优化后完成。
AI 工具主要用于代码实现辅助、文档翻译和文本润色。
完整的设计决策记录、实验数据和迭代过程均保留在项目文档中。

## 许可证

MIT
