"""
@triage_agent 装饰器

将用户的 ReAct 智能体包装为 TRIAGE 三级路由架构。
用户只需一行装饰器，即可自动获得 Token 节省效果。

用法:
    from triage import triage_agent
    
    @triage_agent
    def my_agent(query: str) -> str:
        ...
"""

from __future__ import annotations
import functools
import os
import json
from typing import Optional, Callable

from .agent import TriageAgentWrapper


def triage_agent(
    func: Optional[Callable] = None,
    *,
    db_path: Optional[str] = None,
    llm_config: Optional[dict] = None,
    cold_start: bool = True,
    verbose: bool = False,
):
    """
    将用户的 ReAct 智能体包装为 TRIAGE 三级路由架构。
    
    支持两种用法：
    1. 无参数装饰器：@triage_agent
    2. 有参数装饰器：@triage_agent(db_path="./trajectories.db", verbose=True)
    """
    
    def decorator(f: Callable) -> Callable:
        wrapper = TriageAgentWrapper(
            func=f,
            verbose=verbose,
        )
        
        @functools.wraps(f)
        def wrapped(query: str, **kwargs):
            result = wrapper.run(query, **kwargs)
            return result
            
        wrapped._triage_wrapper = wrapper
        wrapped.get_stats = wrapper.get_stats
        wrapped.get_report = wrapper.get_report
        
        return wrapped
    
    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["triage_agent"]
