import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "s14_mcp_plugin" / "code.py"


def load_lesson(workdir: Path):
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = types.SimpleNamespace(create=None)

    fake_openai.OpenAI = FakeOpenAI
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda override=True: None
    previous = {name: sys.modules.get(name) for name in ("openai", "dotenv")}
    previous_cwd = Path.cwd()
    previous_model = os.environ.get("OPENAI_MODEL_ID")
    spec = importlib.util.spec_from_file_location("s14_openai_test", LESSON)
    module = importlib.util.module_from_spec(spec)
    sys.modules["openai"] = fake_openai
    sys.modules["dotenv"] = fake_dotenv
    try:
        os.chdir(workdir)
        os.environ["OPENAI_MODEL_ID"] = "test-model"
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        if previous_model is None:
            os.environ.pop("OPENAI_MODEL_ID", None)
        else:
            os.environ["OPENAI_MODEL_ID"] = previous_model
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class MCPPluginOpenAITests(unittest.TestCase):
    def test_dynamic_mcp_tool_uses_openai_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            lesson = load_lesson(Path(tmp))
            connect = types.SimpleNamespace(
                type="function_call", call_id="call_connect",
                name="connect_mcp", arguments='{"name":"docs"}',
            )
            get_version = types.SimpleNamespace(
                type="function_call", call_id="call_version",
                name="mcp__docs__get_version", arguments="{}",
            )
            final = types.SimpleNamespace(type="message")
            responses = [
                types.SimpleNamespace(output=[connect], output_text=""),
                types.SimpleNamespace(output=[get_version], output_text=""),
                types.SimpleNamespace(output=[final], output_text="docs API v2.1.0"),
            ]
            tool_names_by_round = []

            def respond(**kwargs):
                tool_names_by_round.append([tool["name"] for tool in kwargs["tools"]])
                return responses.pop(0)

            lesson.client.responses.create = respond
            history = [{"role": "user", "content": "连接docs并查询版本"}]

            result = lesson.agent_loop(history)

            self.assertEqual(result, "docs API v2.1.0")
            self.assertNotIn("mcp__docs__get_version", tool_names_by_round[0])
            self.assertIn("mcp__docs__get_version", tool_names_by_round[1])
            outputs = [item for item in history
                       if isinstance(item, dict)
                       and item.get("type") == "function_call_output"]
            self.assertEqual([item["call_id"] for item in outputs],
                             ["call_connect", "call_version"])
            self.assertIn("API v2.1.0", outputs[1]["output"])
            self.assertTrue(all(tool["type"] == "function"
                                and "parameters" in tool
                                for tool in lesson.assemble_tool_pool()[0]))


if __name__ == "__main__":
    unittest.main()
