from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anthropic API key — loaded from ANTHROPIC_API_KEY env var or .env file
    anthropic_api_key: str = ""

    # Which Claude model the agent uses for reasoning
    model: str = "claude-sonnet-4-6"

    # Hard ceiling on how many reason→act→observe cycles to allow.
    # Higher = more thorough but more expensive. 25 is enough for the fixtures.
    max_iterations: int = 25

    # Per-command timeout inside the sandbox (seconds)
    command_timeout: int = 30

    # Whether to use Docker sandbox. If False, falls back to direct subprocess
    # (unsafe — only for local development when Docker isn't available)
    use_docker: bool = True

    # Truncate tool observations longer than this to avoid blowing the context window
    max_observation_chars: int = 4000


# Module-level singleton — import this everywhere instead of constructing Settings()
settings = Settings()
