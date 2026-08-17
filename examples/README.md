# TRIAGE 示例

## 快速入门

```bash
# 安装
pip install triage-agent

# 配置环境变量
export TRIAGE_API_KEY="sk-xxx"
export TRIAGE_BASE_URL="https://api.openai.com/v1"
export TRIAGE_MODEL="deepseek-v4-flash"
```

## 示例

### quickstart.py
模拟流程演示，无需真实 LLM，展示三级路由工作机制。

### end_to_end.py
端到端示例，使用真实 LLM 演示从冷启动到轨迹复用的完整流程。

## 实验效果

在 237 条安保监控查询的大规模实验中：
- Token 减少：**55.1%**（127,095 → 57,007）
- 直接复用（L1，0 Token）：38.8% 的查询
- 编辑复用（L2，部分 Token）：49.8% 的查询
- 跨领域平均：59.5% Token 减少

详见 `docs/experiments/` 下的实验报告。
