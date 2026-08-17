"""
TRIAGE CLI 入口

支持命令：
- triage run: 运行单次查询
- triage wrap: 包装一个 agent 脚本
- triage report: 查看 Token 节省报告
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Optional

from .model_api import OpenAIChatAPI
from .baseline.orchestrator import TriageOrchestrator


def cmd_run(args: argparse.Namespace):
    """运行单次查询"""
    orchestrator = TriageOrchestrator()
    result = orchestrator.run(
        query=args.query,
        fallback=_make_fallback(args.agent),
    )
    print(result.output)
    if args.verbose:
        print(f"\n[Token] {result.tokens_consumed} tokens (level={result.level})")


def cmd_wrap(args: argparse.Namespace):
    """包装一个 agent 脚本，启动交互式服务"""
    print(f"TRIAGE wrapping: {args.agent}")
    print("Enter queries (Ctrl+D to exit):")
    
    orchestrator = TriageOrchestrator()
    
    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue
        
        result = orchestrator.run(
            query=query,
            fallback=_make_fallback(args.agent),
        )
        print(f"Result: {result.output}")
        print(f"[Token: {result.tokens_consumed} | Level: {result.level}]")
        print()


def cmd_report(args: argparse.Namespace):
    """查看 Token 节省报告"""
    from .retrieval.trajectory_db import TrajectoryDatabase
    db = TrajectoryDatabase()
    stats = db.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _make_fallback(agent_cmd: Optional[str]):
    """从命令行创建一个 fallback 函数"""
    if agent_cmd is None:
        return None
    
    def fallback(query: str) -> str:
        try:
            result = subprocess.run(
                agent_cmd.split() + [query],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: Agent timeout"
        except Exception as e:
            return f"Error: {e}"
    
    return fallback


def main():
    parser = argparse.ArgumentParser(description="TRIAGE CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # triage run
    run_parser = subparsers.add_parser("run", help="Run a single query")
    run_parser.add_argument("--query", "-q", required=True, help="Query string")
    run_parser.add_argument("--agent", "-a", help="Agent command (e.g. 'python my_agent.py')")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Show token stats")
    
    # triage wrap
    wrap_parser = subparsers.add_parser("wrap", help="Wrap an agent script")
    wrap_parser.add_argument("--agent", "-a", required=True, help="Agent command")
    
    # triage report
    report_parser = subparsers.add_parser("report", help="Show token savings report")
    
    args = parser.parse_args()
    
    if args.command == "run":
        cmd_run(args)
    elif args.command == "wrap":
        cmd_wrap(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
