"""Headless Google Antigravity CLI text streaming.

Each call starts one ``agy`` process and yields text delta chunks from
``--output-format stream-json``. The process uses the signed-in
subscription or personal account session. ``GEMINI_API_KEY`` and
``GOOGLE_API_KEY`` are removed from the child environment so an exported
API key is not the credential.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

# None means the CLI's configured default model, so the library works with any
# sign-in. Pass model="gemini-3.8-flash-high" or another model name to pin one.
DEFAULT_MODEL: str | None = None
STREAM_LIMIT = 8 * 1024 * 1024

# API key variables that could bypass the signed-in session are removed.
_API_KEY_ENV = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_AI_API_KEY",
)

# Parent agent session variables to prevent child session collisions.
_AGENT_SESSION_ENV = (
    "ANTIGRAVITY_AGENT",
    "ANTIGRAVITY_CONVERSATION_ID",
    "ANTIGRAVITY_TRAJECTORY_ID",
    "ANTIGRAVITY_SOURCE_METADATA",
)

# The system prompt owns the task. The user message is the input to that task.
_TEXT_ONLY_RULE = (
    "Do not call tools, read or edit files, browse the web, or run commands. "
    "Obey the system prompt. The user message is the input to that task. "
    "When the system prompt asks for a specification or a formatted block instead of a direct answer, produce only that specification."
)


class AntigravityError(RuntimeError):
    """The Google Antigravity CLI is missing, or a turn failed before any text."""


AgyError = AntigravityError


def find_agy_bin() -> str:
    """Return the ``agy`` executable from ``PATH`` or a known install path."""
    found = shutil.which("agy")
    if found and os.path.isfile(found) and os.access(found, os.X_OK):
        return found
    for location in (
        "~/.local/bin/agy",
        "~/.gemini/antigravity-cli/bin/agy",
        "~/.antigravity/bin/agy",
        "~/.antigravity-ide/antigravity-ide/bin/agy",
        "/usr/local/bin/agy",
    ):
        fallback = os.path.expanduser(location)
        if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
            return fallback
    raise AntigravityError(
        "The agy CLI was not found. Install Google Antigravity CLI, then sign in with: agy"
    )


find_antigravity_bin = find_agy_bin


def subscription_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process that keeps the ``agy`` login session."""
    env = dict(os.environ if base is None else base)
    for name in _API_KEY_ENV:
        env.pop(name, None)
    for name in _AGENT_SESSION_ENV:
        env.pop(name, None)
    env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "1"
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if not isinstance(text, str):
                    text = item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def format_prompt(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
) -> tuple[str, str]:
    """Split chat messages into a user prompt and a system prompt.

    A lone user turn is left unlabeled so the model sees that text exactly.
    Earlier turns are labeled when history is present. The explicit ``system``
    argument is the system prompt. System messages already in ``messages`` stay
    in the user prompt, so a generic identity does not outrank the task.
    """
    system_parts: list[str] = []
    if isinstance(system, str) and system.strip():
        system_parts.append(system.strip())
    standing: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages or ():
        role = str(message.get("role") or "user").strip().lower()
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            standing.append(content)
        elif role == "assistant":
            turns.append(("Assistant", content))
        else:
            turns.append(("User", content))
    if isinstance(follow_up, str) and follow_up.strip():
        turns.append(("User", follow_up.strip()))
    system_parts.append(_TEXT_ONLY_RULE)
    if not turns and not standing:
        raise AntigravityError("No prompt to send.")
    if len(turns) == 1 and turns[0][0] == "User":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"{role}:\n{content}" for role, content in turns)
    if standing:
        context = "\n\n".join(standing)
        prompt = f"{context}\n\n{prompt}" if prompt else context
    return prompt, "\n\n".join(system_parts)


def compose_prompt(prompt: str, system_prompt: str) -> str:
    """Combine user prompt and system prompt for the agy CLI."""
    if system_prompt and system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{prompt}"
    return prompt


def build_command(
    agy_bin: str,
    prompt: str,
    model: str | None = None,
    effort: str | None = None,
) -> list[str]:
    """Argv for one text-only headless turn. No API key and no tool approval."""
    command = [
        agy_bin,
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
    ]
    chosen_model = model.strip() if isinstance(model, str) and model.strip() else DEFAULT_MODEL
    if chosen_model:
        command.extend(("--model", chosen_model))
    if isinstance(effort, str) and effort.strip():
        command.extend(("--effort", effort.strip()))
    command.extend(("--print", prompt))
    return command


def parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse one stdout line from ``--output-format stream-json``."""
    text = line.strip()
    if not text:
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return event
    return None


def text_from_event(event: Mapping[str, Any]) -> str | None:
    """Return a streamed text chunk. Other event types contribute no answer text."""
    if event.get("event") != "step_update":
        return None
    step_update = event.get("step_update")
    if not isinstance(step_update, dict):
        return None
    if step_update.get("step_type") != "agent_response":
        return None
    delta = step_update.get("text_delta")
    if isinstance(delta, str) and delta:
        return delta
    return None


def result_text_from_event(event: Mapping[str, Any]) -> str | None:
    """Return text from a whole ``result`` event.

    This is the fallback if no text delta was streamed.
    """
    if event.get("event") != "result":
        return None
    result = event.get("result")
    if not isinstance(result, dict) or result.get("status") == "ERROR":
        return None
    response = result.get("response")
    if isinstance(response, str) and response:
        return response
    return None


def _completion_workspace() -> str:
    """Stable directory outside the caller's project, with no git root of its own."""
    path = os.path.join(tempfile.gettempdir(), "antigravitystreaming-workspace")
    os.makedirs(path, exist_ok=True)
    return path


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        return


async def _next_line(stream: asyncio.StreamReader, cancel_event: asyncio.Event | None) -> bytes | None:
    """Read one stdout line. ``None`` means the caller cancelled."""
    read_task = asyncio.create_task(stream.readline())
    if cancel_event is None:
        return await read_task
    if cancel_event.is_set():
        await _cancel_task(read_task)
        return None
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {read_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        await _cancel_task(read_task)
        await _cancel_task(cancel_task)
        raise
    if read_task in done and not read_task.cancelled():
        await _cancel_task(cancel_task)
        return read_task.result()
    await _cancel_task(read_task)
    await _cancel_task(cancel_task)
    return None


async def _read_stderr(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    chunks: list[str] = []
    size = 0
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            size += len(text)
            if size > 16000:
                chunks = chunks[-20:]
                size = sum(len(item) for item in chunks)
    except asyncio.CancelledError:
        return "".join(chunks)[-8000:]
    return "".join(chunks)[-8000:]


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception:
                process.kill()
        try:
            await process.wait()
        except Exception:
            return


async def stream_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    agy_bin: str | None = None,
    antigravity_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
):
    """Yield text chunks from one signed-in Google Antigravity CLI turn.

    Closing the generator, or setting ``cancel_event``, stops the CLI process.
    ``model`` defaults to :data:`DEFAULT_MODEL` (the CLI's configured model).
    The working directory is a temporary folder, and this call does not resume
    an earlier session.
    """
    prompt, system_prompt = format_prompt(messages, follow_up=follow_up, system=system)
    full_prompt = compose_prompt(prompt, system_prompt)
    executable = agy_bin or antigravity_bin or find_agy_bin()
    work_dir = _completion_workspace()
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task | None = None
    produced_text = False
    try:
        command = build_command(executable, full_prompt, model=model, effort=effort)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=subscription_env(),
            start_new_session=True,
            limit=STREAM_LIMIT,
        )
        if process.stdout is None:
            raise AntigravityError("agy did not provide an output stream")
        stderr_task = asyncio.create_task(_read_stderr(process.stderr))
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            line = await _next_line(process.stdout, cancel_event)
            if line is None:
                return
            if line == b"":
                break
            event = parse_stream_line(line.decode("utf-8", errors="replace"))
            if not event:
                continue
            if event.get("event") == "result":
                if cancel_event is not None and cancel_event.is_set():
                    return
                res = event.get("result")
                if isinstance(res, dict):
                    if res.get("status") == "ERROR" or res.get("error"):
                        message = res.get("error") or "Google Antigravity CLI reported an error."
                        raise AntigravityError(str(message))
                    if not produced_text:
                        text = res.get("response")
                        if isinstance(text, str) and text:
                            produced_text = True
                            yield text
                continue
            chunk = text_from_event(event)
            if chunk:
                produced_text = True
                yield chunk
                continue
            if not produced_text:
                chunk = result_text_from_event(event)
                if chunk:
                    produced_text = True
                    yield chunk
        returncode = await process.wait()
        stderr_text = ""
        if stderr_task is not None:
            stderr_text = (await stderr_task).strip()
            stderr_task = None
        if returncode not in (0, None) and not produced_text:
            detail = stderr_text or f"agy exited with code {returncode}"
            raise AntigravityError(detail)
    finally:
        await _stop_process(process)
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass


async def collect_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    agy_bin: str | None = None,
    antigravity_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
) -> str:
    """Run :func:`stream_text` and return the joined text."""
    chunks: list[str] = []
    async for chunk in stream_text(
        messages,
        follow_up=follow_up,
        system=system,
        model=model,
        effort=effort,
        agy_bin=agy_bin,
        antigravity_bin=antigravity_bin,
        cancel_event=cancel_event,
    ):
        chunks.append(chunk)
    return "".join(chunks)
