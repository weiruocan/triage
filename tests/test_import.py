"""TRIAGE 快速单元测试（不加载模型，仅验证导入和逻辑）"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "triage"))


def test_core_imports():
    """验证核心模块导入"""
    from triage import (
        triage_agent, TriageOrchestrator, OpenAIChatAPI,
        ExecutionFlow, SkillExtractor, __version__,
    )
    assert triage_agent is not None
    assert __version__ == "0.2.0"
    print(f"  [PASS] 核心导入 v{__version__}")


def test_retrieval_imports():
    """验证检索模块导入"""
    from triage.retrieval import (
        ExperienceNode, ExperienceTrajectory, ExperienceEditor,
        reference_generate, no_change, execute_edit_strategy, EditResult,
    )
    assert ExperienceNode is not None
    assert reference_generate is not None
    print(f"  [PASS] 检索模块")


def test_skills_imports():
    """验证技能模块导入"""
    from triage.skills import (
        ExecutionFlow, register_execution_flow, get_execution_flow,
        SkillExtractor, register_all_skills,
    )
    assert ExecutionFlow is not None
    assert SkillExtractor is not None
    print(f"  [PASS] 技能模块")


def test_router_imports():
    from triage.router import LLMRouter
    assert LLMRouter is not None
    print(f"  [PASS] 路由模块")


def test_baseline_imports():
    from triage.baseline import TriageOrchestrator, StepExecutor
    assert TriageOrchestrator is not None
    print(f"  [PASS] 基线模块")


def test_cli_import():
    from triage.cli import main
    assert main is not None
    print(f"  [PASS] CLI 模块")


def test_edit_strategies():
    """验证编辑策略逻辑"""
    from triage.retrieval.edit_strategies import no_change, EditResult
    
    node = {"action_name": "execute_sql", "tool_input": {"sql": "SELECT 1"}}
    result = no_change(node)
    assert result.success
    assert result.edited_node is not None
    assert result.edited_node["edited"] == False
    assert result.token_cost == 0
    print(f"  [PASS] 编辑策略 - no_change (0 Token)")


def test_experience_trajectory():
    """验证轨迹数据模型"""
    from triage.retrieval.experience_trajectory import ExperienceNode, ExperienceTrajectory
    
    node = ExperienceNode(
        step_index=0,
        action_name="execute_sql",
        description="查传感器在线率",
        node_type="tool",
        tool_input={"sql": "SELECT 1"},
        token_input=100,
        token_output=50,
    )
    assert node.token_cost == 150
    assert node.is_zero_token == True  # tool 类型
    
    traj = ExperienceTrajectory(query="test", category="test")
    traj.add_node(node)
    assert traj.total_token_cost == 150
    assert traj.zero_token_ratio == 1.0
    print(f"  [PASS] 轨迹数据模型")


def test_execution_flow():
    """验证 ExecutionFlow 注册表"""
    from triage.skills.execution_flow import (
        ExecutionFlow, register_execution_flow, get_execution_flow, list_execution_flows,
    )
    
    # 注册一个测试 Skill
    flow = ExecutionFlow(
        skill_name="test_skill",
        description="测试 Skill",
        input_schema={},
        output_schema={},
        executor=lambda **kwargs: {"success": True},
        estimated_token_cost=0,
    )
    register_execution_flow(flow)
    assert get_execution_flow("test_skill") is not None
    assert "test_skill" in list_execution_flows()
    print(f"  [PASS] ExecutionFlow 注册表")


def test_skill_extractor():
    """验证 Skill 提取器"""
    from triage.skills.skill_extractor import SkillExtractor
    
    extractor = SkillExtractor(min_frequency=2)
    
    traj1 = ('{"query": "sensor online rate", "nodes": ['
             '{"action_name": "execute_sql", '
             '"tool_input": {"sql": "SELECT * FROM sensors WHERE type = \'temperature\'"}}]}')
    traj2 = ('{"query": "camera online rate", "nodes": ['
             '{"action_name": "execute_sql", '
             '"tool_input": {"sql": "SELECT * FROM sensors WHERE type = \'camera\'"}}]}')
    
    extractor.analyze_trajectory(traj1, 1)
    extractor.analyze_trajectory(traj2, 2)
    
    assert len(extractor._patterns) > 0
    print(f"  [PASS] Skill 提取器 ({len(extractor._patterns)} 个模式)")


def test_experience_editor():
    """验证轨迹编辑器"""
    from triage.retrieval.experience_editor import ExperienceEditor
    editor = ExperienceEditor()
    assert editor is not None
    assert hasattr(editor, "edit_trajectory")
    print(f"  [PASS] 轨迹编辑器")


if __name__ == "__main__":
    print("TRIAGE 单元测试（不加载模型）")
    print("=" * 40)
    
    tests = [
        test_core_imports, test_retrieval_imports, test_skills_imports,
        test_router_imports, test_baseline_imports, test_cli_import,
        test_edit_strategies, test_experience_trajectory,
        test_execution_flow, test_skill_extractor, test_experience_editor,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {test.__name__}: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*40}")
    print(f"结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print("全部通过 ✅")
    else:
        print(f"{len(tests) - passed} 个失败 ❌")
