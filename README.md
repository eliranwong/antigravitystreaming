# antigravitystreaming

Stream text from the Google Antigravity CLI (`agy`). This library is a text-completion alternative to [agentmake](https://github.com/eliranwong/agentmake): you pass chat messages, and it yields response text as the CLI produces it.

Authentication is the signed-in session managed by `agy` (under `~/.gemini/antigravity-cli`). The child process does not receive `GEMINI_API_KEY` or `GOOGLE_API_KEY`. No API key is read or sent.

## Install

```bash
pip install antigravitystreaming
```

Requires the `agy` executable on `PATH`, at `~/.local/bin/agy`, or at `~/.gemini/antigravity-cli/bin/agy`, and a completed `agy` sign-in.

## Stream

```python
import asyncio
from antigravitystreaming import stream_text

async def main():
    messages = [
        {"role": "system", "content": "You are a concise Bible study assistant."},
        {"role": "user", "content": "What is John 3:16 about?"},
        {"role": "assistant", "content": "It states God's love for the world."},
    ]
    async for chunk in stream_text(messages, follow_up="Quote it briefly."):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

`messages` use `role` and `content`, the same shape agentmake accepts. `follow_up` is the newest user turn. An explicit `system` argument or system messages in `messages` are combined with the prompt under text-only guidance so instructions stay intact. Each call is a new headless process: the library does not resume a previous Antigravity session.

The process is text-only. It runs non-interactively with `--output-format stream-json`, `--dangerously-skip-permissions`, and `--disable-slash-commands`, in a temporary working directory outside the caller's project. Cancel an in-flight stream by setting the `asyncio.Event` passed as `cancel_event`.

`model` defaults to the CLI's configured model (e.g. `Gemini 3.8 Flash (High)`), so the library follows whatever `agy` is signed in and set up to run. Pass `model="gemini-3.8-flash-high"` or another model name to pin one. Pass `effort="high"` (or low, medium, xhigh, max) to configure reasoning effort.

`collect_text` returns the whole reply:

```python
from antigravitystreaming import collect_text

text = await collect_text(follow_up="Reply with exactly: pong")
```
