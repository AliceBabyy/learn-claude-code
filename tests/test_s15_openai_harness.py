import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "s15_integrated_harness" / "code.py"


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
    spec = importlib.util.spec_from_file_location("s15_openai_test", LESSON)
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


def response(output, text=""):
    return types.SimpleNamespace(
        output=output,
        output_text=text,
        status="completed",
        incomplete_details=None,
    )


class IntegratedHarnessOpenAITests(unittest.TestCase):
    def test_main_loop_and_compaction_use_openai_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            lesson = load_lesson(Path(tmp))
            lesson.update_context = lambda context, messages: context
            lesson.remember_after_turn = lambda messages: None
            reasoning = types.SimpleNamespace(type="reasoning", id="rs_1")
            call = types.SimpleNamespace(
                type="function_call", call_id="call_list",
                name="list_tasks", arguments="{}",
            )
            final = types.SimpleNamespace(type="message", role="assistant", content=[])
            queue = [response([reasoning, call]), response([final], "完成")]
            lesson.client.responses.create = lambda **_: queue.pop(0)
            history = [{"role": "user", "content": "列出任务"}]

            lesson.agent_loop(history, {}, "列出任务")

            self.assertIs(history[1], reasoning)
            self.assertIs(history[2], call)
            self.assertEqual(history[3]["type"], "function_call_output")
            self.assertEqual(history[3]["call_id"], "call_list")
            self.assertTrue(all(tool["type"] == "function"
                                and "parameters" in tool
                                for tool in lesson.assemble_tool_pool()[0]))

            lesson.TOOL_RESULTS_DIR = Path(tmp) / "tool-results"
            lesson.PERSIST_THRESHOLD = 20
            fresh = [
                {"role": "user", "content": "读取"},
                {"type": "reasoning", "id": "r2"},
                {"type": "function_call", "call_id": "c2"},
                {"type": "function_call_output", "call_id": "c2",
                 "output": "X" * 100},
            ]
            lesson.function_output_budget(fresh, max_bytes=10)
            self.assertTrue(list(lesson.TOOL_RESULTS_DIR.glob("*")))
            preview = fresh[-1]["output"]
            lesson.micro_compact(fresh)
            self.assertEqual(fresh[-1]["output"], preview)

    def test_subagent_returns_only_final_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            lesson = load_lesson(Path(tmp))
            final = types.SimpleNamespace(type="message", role="assistant", content=[])
            seen = []

            def respond(**kwargs):
                seen.append(list(kwargs["input"]))
                return response([final], "子任务结论")

            lesson.client.responses.create = respond
            self.assertEqual(lesson.spawn_subagent("检查项目"), "子任务结论")
            self.assertEqual(seen[0], [{"role": "user", "content": "检查项目"}])
            self.assertNotIn("task", [tool["name"] for tool in lesson.SUB_TOOLS])


if __name__ == "__main__":
    unittest.main()
