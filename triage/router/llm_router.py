"""LLMRouter"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Optional
from ..model_api import OpenAIChatAPI

@dataclass
class RouterDecision:
    level: str; matched_trajectory_id: Optional[str] = None; edit_nodes: list[dict] = field(default_factory=list)
    def to_dict(self) -> dict:
        """转换为字典"""
        return {"level": self.level, "matched_trajectory_id": self.matched_trajectory_id,
                "edit_nodes": self.edit_nodes, "reasoning": self.reasoning, "confidence": self.confidence}

    reasoning: str = ""; confidence: float = 0.0

ROUTER_PROMPT = """Classify the new query into: one (direct reuse, 0 Token), two (edit reuse, edit nodes only), three (full ReAct).

Trajectories:
{trajectory_summaries}

New query: {query}

Output JSON: {{"level":"one"/"two"/"three","matched_trajectory_id":"id or null","edit_nodes":[{{"step_index":0,"edit_type":"reference_generate"/"no_change","reason":"..."}}],"reasoning":"...","confidence":0-1}}"""

class LLMRouter:
    def __init__(self, model_api: OpenAIChatAPI, top_k: int = 3): self.model_api = model_api; self.top_k = top_k
    def _summarize(self, trajectories):
        if not trajectories: return "No trajectories found."
        ss = []
        for i, t in enumerate(trajectories[:self.top_k]):
            traj = t.get("trajectory", {})
            if isinstance(traj, str):
                try: traj = json.loads(traj)
                except: traj = {}
            q = traj.get("query", t.get("query", "unknown"))
            nodes = traj.get("nodes", [])
            ns = "\n".join([f"  Step {n.get('step_index',0)}: {n.get('action_name','?')} [{n.get('node_type','?')}]" for n in nodes[:5]])
            ss.append(f"[{i}] ID: {traj.get('trajectory_id', t.get('id','?'))}\n    Query: {q[:150]}\n    Steps: {len(nodes)}\n    Nodes:\n{ns}")
        return "\n\n".join(ss)
    def _parse(self, text: str) -> RouterDecision:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return RouterDecision(level=d.get("level","three"), matched_trajectory_id=d.get("matched_trajectory_id"),
                    edit_nodes=d.get("edit_nodes",[]), reasoning=d.get("reasoning",""), confidence=d.get("confidence",0.0))
            except: pass
        if "level 1" in text.lower() or "level one" in text.lower(): return RouterDecision(level="one")
        if "level 2" in text.lower() or "level two" in text.lower(): return RouterDecision(level="two")
        return RouterDecision(level="three")
    def route(self, query: str, retrieved: list) -> RouterDecision:
        for t in retrieved:
            traj = t.get("trajectory", {})
            if isinstance(traj, str):
                try: traj = json.loads(traj)
                except: continue
            eq = traj.get("query", t.get("query", ""))
            if eq.strip().lower() == query.strip().lower():
                return RouterDecision(level="one", matched_trajectory_id=traj.get("trajectory_id", str(t.get("id",""))),
                    edit_nodes=[], reasoning="Exact match", confidence=1.0)
        prompt = ROUTER_PROMPT.format(trajectory_summaries=self._summarize(retrieved), query=query)
        response = self.model_api.chat(prompt, temperature=0.3)
        return self._parse(response)
    def route_no_llm(self, query: str, retrieved: list, threshold: float = 0.85) -> RouterDecision:
        if not retrieved: return RouterDecision(level="three", reasoning="No matches")
        best = retrieved[0]; sim = best.get("similarity", 0.0)
        traj = best.get("trajectory", {})
        if isinstance(traj, str):
            try: traj = json.loads(traj)
            except: traj = {}
        tid = traj.get("trajectory_id", str(best.get("id","")))
        eq = traj.get("query", best.get("query",""))
        if eq.strip().lower() == query.strip().lower(): return RouterDecision(level="one", matched_trajectory_id=tid, reasoning="Exact match", confidence=1.0)
        if sim >= threshold: return RouterDecision(level="two", matched_trajectory_id=tid, reasoning=f"Similarity {sim:.3f}", confidence=sim)
        return RouterDecision(level="three", matched_trajectory_id=tid, reasoning=f"Similarity {sim:.3f} < {threshold}", confidence=sim)
