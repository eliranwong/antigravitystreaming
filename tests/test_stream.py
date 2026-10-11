import unittest
from unittest import mock

from antigravitystreaming.stream import (
    DEFAULT_MODEL,
    AgyError,
    AntigravityError,
    build_command,
    compose_prompt,
    find_agy_bin,
    format_prompt,
    parse_stream_line,
    result_text_from_event,
    subscription_env,
    text_from_event,
)


class FormatPromptTests(unittest.TestCase):
    def test_single_user_turn_is_unlabeled(self):
        prompt, system = format_prompt(
            [{"role": "system", "content": "You are BibleMate AI."}],
            follow_up="List two tools.",
        )
        self.assertIn("You are BibleMate AI.", prompt)
        self.assertTrue(prompt.endswith("List two tools."))
        self.assertNotIn("You are BibleMate AI.", system)
        self.assertNotIn("List two tools.", system)
        self.assertIn("Obey the system prompt.", system)
        self.assertNotIn("User:", prompt)

    def test_history_keeps_the_latest_request_intact(self):
        prompt, system = format_prompt(
            [
                {"role": "system", "content": "Role."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
            follow_up="Second question",
            system="Extra role",
        )
        self.assertIn("Role.", prompt)
        self.assertIn("User:\nFirst question", prompt)
        self.assertIn("Assistant:\nFirst answer", prompt)
        self.assertTrue(prompt.endswith("User:\nSecond question"))
        self.assertIn("Extra role", system)
        self.assertNotIn("Role.", system)
        self.assertNotIn("Second question", system)

    def test_content_blocks_join_text(self):
        prompt, _system = format_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"text": "there"}]}],
        )
        self.assertEqual(prompt, "Hello\nthere")

    def test_empty_prompt_raises(self):
        with self.assertRaises(AntigravityError):
            format_prompt([])

    def test_compose_prompt(self):
        composed = compose_prompt("User query", "System instruction")
        self.assertEqual(composed, "System instruction\n\nUser query")
        self.assertEqual(compose_prompt("User query", ""), "User query")


class EventTests(unittest.TestCase):
    def test_text_delta_chunk(self):
        event = parse_stream_line(
            '{"event":"step_update","step_update":{"conversation_id":"c1",'
            '"step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"pong"}}'
        )
        self.assertEqual(text_from_event(event), "pong")

    def test_user_input_step_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"event":"step_update","step_update":{"conversation_id":"c1",'
            '"step_index":0,"state":"DONE","step_type":"user_input"}}'
        )
        self.assertIsNone(text_from_event(event))

    def test_tool_event_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"event":"step_update","step_update":{"conversation_id":"c1",'
            '"step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"view_file"}}'
        )
        self.assertIsNone(text_from_event(event))

    def test_result_event_fallback(self):
        event = parse_stream_line(
            '{"event":"result","result":{"conversation_id":"c1","status":"SUCCESS","response":"pong\\n"}}'
        )
        self.assertEqual(result_text_from_event(event), "pong\n")

    def test_result_error_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"event":"result","result":{"conversation_id":"c1","status":"ERROR","error":"bad model"}}'
        )
        self.assertIsNone(result_text_from_event(event))

    def test_non_json_is_ignored(self):
        self.assertIsNone(parse_stream_line("plain log line"))


class CommandTests(unittest.TestCase):
    def test_command_is_text_only_and_has_no_api_key(self):
        command = build_command("/usr/bin/agy", "Be brief.")
        joined = " ".join(command)
        self.assertNotIn("GEMINI_API_KEY", joined)
        self.assertNotIn("GOOGLE_API_KEY", joined)
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("--disable-slash-commands", command)
        self.assertIn("--print", command)
        self.assertIn("Be brief.", command)
        self.assertNotIn("--model", command)

    def test_explicit_model_is_pinned(self):
        command = build_command("/usr/bin/agy", "Be brief.", model="gemini-3.8-flash-high")
        self.assertIn("--model", command)
        self.assertIn("gemini-3.8-flash-high", command)

    def test_explicit_effort_is_pinned(self):
        command = build_command("/usr/bin/agy", "Be brief.", effort="high")
        self.assertIn("--effort", command)
        self.assertIn("high", command)

    def test_default_model_is_none(self):
        self.assertIsNone(DEFAULT_MODEL)

    def test_agy_error_alias(self):
        self.assertIs(AgyError, AntigravityError)

    def test_subscription_env_drops_api_keys_and_agent_vars(self):
        env = subscription_env(
            {
                "GEMINI_API_KEY": "sk-secret",
                "GOOGLE_API_KEY": "sk-secret-2",
                "ANTIGRAVITY_AGENT": "1",
                "ANTIGRAVITY_CONVERSATION_ID": "conv-1",
                "PATH": "/usr/bin",
                "HOME": "/Users/me",
            }
        )
        self.assertNotIn("GEMINI_API_KEY", env)
        self.assertNotIn("GOOGLE_API_KEY", env)
        self.assertNotIn("ANTIGRAVITY_AGENT", env)
        self.assertNotIn("ANTIGRAVITY_CONVERSATION_ID", env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["AGY_CLI_DISABLE_AUTO_UPDATE"], "1")
        self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")

    def test_find_agy_bin_uses_path(self):
        with mock.patch("antigravitystreaming.stream.shutil.which", return_value="/opt/agy"):
            with mock.patch("antigravitystreaming.stream.os.path.isfile", return_value=True):
                with mock.patch("antigravitystreaming.stream.os.access", return_value=True):
                    self.assertEqual(find_agy_bin(), "/opt/agy")

    def test_find_agy_bin_uses_known_install_fallback(self):
        with mock.patch("antigravitystreaming.stream.shutil.which", return_value=None):
            with mock.patch("antigravitystreaming.stream.os.path.expanduser", side_effect=lambda p: p):
                with mock.patch("antigravitystreaming.stream.os.path.isfile", return_value=True):
                    with mock.patch("antigravitystreaming.stream.os.access", return_value=True):
                        self.assertEqual(find_agy_bin(), "~/.local/bin/agy")

    def test_find_agy_bin_errors_when_missing(self):
        with mock.patch("antigravitystreaming.stream.shutil.which", return_value=None):
            with mock.patch("antigravitystreaming.stream.os.path.expanduser", side_effect=lambda p: p):
                with mock.patch("antigravitystreaming.stream.os.path.isfile", return_value=False):
                    with self.assertRaises(AntigravityError):
                        find_agy_bin()


if __name__ == "__main__":
    unittest.main()
