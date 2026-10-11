"""Stream text from the signed-in Google Antigravity CLI.

The CLI session managed by Google Antigravity CLI (``agy``) is the credential.
This package does not read or send an API key.
"""

from antigravitystreaming.stream import (
    DEFAULT_MODEL,
    AgyError,
    AntigravityError,
    build_command,
    collect_text,
    compose_prompt,
    find_agy_bin,
    find_antigravity_bin,
    format_prompt,
    parse_stream_line,
    result_text_from_event,
    stream_text,
    subscription_env,
    text_from_event,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "AgyError",
    "AntigravityError",
    "build_command",
    "collect_text",
    "compose_prompt",
    "find_agy_bin",
    "find_antigravity_bin",
    "format_prompt",
    "parse_stream_line",
    "result_text_from_event",
    "stream_text",
    "subscription_env",
    "text_from_event",
]
