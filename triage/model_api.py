"""
model_api — LLM API 封装

统一调用接口，配置从 config.yaml 读取。
支持重试、超时、Token 统计。
"""

from __future__ import annotations
import os
import time
from typing import Optional


def load_config() -> dict:
    """加载配置，支持环境变量覆盖（环境变量优先）"""
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    config_path = os.path.abspath(config_path)
    config = {}
    # 环境变量覆盖（优先级更高）
    env_overrides = {
        "base_url": os.getenv("TRIAGE_BASE_URL"),
        "api_key": os.getenv("TRIAGE_API_KEY"),
        "model": os.getenv("TRIAGE_MODEL"),
    }
    llm = config.get("llm", {})
    for key, env_val in env_overrides.items():
        if env_val is not None:
            llm[key] = env_val
    config["llm"] = llm
    return config


class OpenAIChatAPI:
    """OpenAI 兼容 API 封装（带重试和超时）"""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        llm_config = config.get("llm", {})
        self.base_url = llm_config.get("base_url", "https://api.openai.com/v1")
        self.api_key = llm_config.get("api_key", "")
        self.model = llm_config.get("model", "gpt-4")
        self.temperature = llm_config.get("temperature", 0.5)
        self.max_tokens = llm_config.get("max_tokens", 8192)
        self.timeout = llm_config.get("timeout", 60)
        self.max_retries = llm_config.get("max_retries", 3)
        self.retry_delay = llm_config.get("retry_delay", 5)

        # Token 统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0

    def chat(self, prompt: str, system_prompt: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """
        单次 LLM 调用（带重试）

        Args:
            prompt: 用户消息
            system_prompt: 系统提示词
            temperature: 温度
            max_tokens: 最大 Token 数

        Returns:
            模型回复文本

        Raises:
            RuntimeError: 重试耗尽后仍失败
        """
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Token 统计
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_calls += 1

                return data["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout as e:
                last_error = f"Timeout (attempt {attempt + 1}/{self.max_retries})"
                self.failed_calls += 1
            except requests.exceptions.ConnectionError as e:
                last_error = f"ConnectionError (attempt {attempt + 1}/{self.max_retries}): {str(e)[:60]}"
                self.failed_calls += 1
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                last_error = f"HTTP {status} (attempt {attempt + 1}/{self.max_retries})"
                self.failed_calls += 1
                # 4xx 错误不重试
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise RuntimeError(f"API 请求失败 (HTTP {status}): {e.response.text[:200]}")
            except Exception as e:
                last_error = f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {str(e)[:60]}"
                self.failed_calls += 1

            # 重试前等待
            if attempt < self.max_retries - 1:
                wait = self.retry_delay * (attempt + 1)
                time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败（重试 {self.max_retries} 次后）: {last_error}")

    def get_token_stats(self) -> dict:
        """获取累计 Token 统计"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
        }

    def reset_stats(self):
        """重置 Token 统计"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0
