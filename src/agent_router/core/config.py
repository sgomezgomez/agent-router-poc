"""Configuration management using Pydantic Settings."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict
from agent_router.core.types import JsonObject
import yaml
from pathlib import Path


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    enabled: bool | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_mode: str | None = None  # chat_completions | responses, for OpenAI-compatible APIs

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None

    # Extended thinking (for models that support it like o1)
    thinking_budget: int | None = None  # Tokens allocated for thinking

    # Request parameters
    timeout: int | None = None
    max_retries: int | None = None
    retry_delay: float | None = None


class MongoDBConfig(BaseModel):
    """MongoDB configuration."""

    uri: str
    database: str
    username: str | None
    password: str | None
    max_pool_size: int
    min_pool_size: int


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str
    format: str
    file: str | None = None


class Settings(BaseSettings):
    """Global settings loaded from .env and YAML files."""

    # MongoDB
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)

    # Brave Search (MCP)
    brave_api_key: str | None = None
    
    # LLM Providers (loaded from .env, overridden by YAML)
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    grok_api_key: str | None = None
    grok_base_url: str | None = None
    lm_studio_base_url: str

    # Fallback configuration (last resort when all retries fail or no provider/model)
    fallback_llm_provider: str
    fallback_llm_model: str
    fallback_llm_api_mode: str
    fallback_temperature: float  # Lower temp for more reliable fallback
    fallback_max_tokens: int
    fallback_top_p: float | None = None
    fallback_top_k: int | None = None
    fallback_thinking_budget: int | None = None
    fallback_thinking_effort: str | None = None
    fallback_enable_thinking: bool | None = None
    adapter_debug_trace_default: bool = False
    llm_debug_logging: bool = False

    # LLM retry configuration
    llm_retry_max_retries: int
    llm_retry_base_delay: float
    llm_retry_max_delay: float
    llm_retry_exponential_base: float

    # Logging
    logging: LoggingConfig

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    def get_llm_providers_config(self) -> Dict[str, LLMProviderConfig]:
        """Load LLM provider configurations from YAML.

        Returns:
            Dictionary mapping provider names to their configurations
        """
        config_path = Path("config/llm_providers.yaml")
        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f)
                return {
                    name: LLMProviderConfig(**cfg)
                    for name, cfg in yaml_config.get("providers", {}).items()
                }

        # Fallback to defaults from .env
        providers = {}
        if self.openai_api_key:
            providers["openai"] = LLMProviderConfig(
                api_key=self.openai_api_key
            )
        if self.gemini_api_key:
            providers["gemini"] = LLMProviderConfig(
                api_key=self.gemini_api_key
            )
        if self.grok_api_key:
            providers["grok"] = LLMProviderConfig(
                api_key=self.grok_api_key,
                base_url=self.grok_base_url
            )
        providers["lm_studio"] = LLMProviderConfig(
            base_url=self.lm_studio_base_url
        )
        return providers

    def get_fallback_config(self) -> JsonObject:
        """Get fallback LLM configuration for last-resort retries.

        This is used when all normal retries with the primary provider fail.
        The fallback typically uses a more conservative configuration (lower temp)
        and a reliable provider (usually local LM Studio).

        Returns:
            Dictionary with fallback provider, model, and parameters
        """
        return {
            "provider": self.fallback_llm_provider,
            "model": self.fallback_llm_model,
            "temperature": self.fallback_temperature,
            "max_tokens": self.fallback_max_tokens,
            "top_p": self.fallback_top_p,
            "top_k": self.fallback_top_k,
            "thinking_budget": self.fallback_thinking_budget,
            "thinking_effort": self.fallback_thinking_effort,
            "enable_thinking": self.fallback_enable_thinking,
        }


class AgentConfig(BaseModel):
    """Configuration for an agent persona loaded from YAML."""

    name: str
    background: str
    goal: str
    task: str | None = None
    guidelines: str | None = None
    examples: str | None = None
    guardrails: str | None = None
    message_window: str = "LAST_10_TURNS"
    max_context_tokens: int = 4000

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentConfig":
        """Load agent config from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            AgentConfig instance
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            # Assumes YAML has a single top-level key (agent name)
            agent_name = list(data.keys())[0]
            config = data[agent_name]
            config["name"] = agent_name
            return cls(**config)
