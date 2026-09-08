import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL_ID", "test-model")

spec = importlib.util.spec_from_file_location(
    "s08_openai", ROOT / "s08_context_compact" / "code.py"
)
s08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s08)


class FakeSdkItem:
    def model_dump(self, mode="python"):
        return {"type": "reasoning", "id": "sdk-r1", "mode": mode}


class OpenAICompactionTests(unittest.TestCase):
    def test_openai_compaction_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compactor = s08.ContextCompactor(
                None, "fake", root / "transcripts", root / "results"
            )
            compactor.LARGE_RESULT_CHAR_LIMIT = 20

            fresh = [
                {"role": "user", "content": "读取大文件"},
                {"type": "reasoning", "id": "r1"},
                {"type": "function_call", "call_id": "c1"},
                {"type": "function_call_output", "call_id": "c1", "output": "X" * 100},
            ]
            compactor.tool_result_budget(fresh, max_chars=10)
            saved = list((root / "results").glob("*"))
            self.assertEqual(saved[0].read_text(encoding="utf-8"), "X" * 100)

            preview = fresh[-1]["output"]
            compactor.micro_compact(fresh)
            self.assertEqual(fresh[-1]["output"], preview)

            messages = [
                {"role": "user", "content": "start"},
                {"type": "message", "id": "m0"},
                {"type": "message", "id": "m1"},
                {"type": "message", "id": "m2"},
                {"type": "reasoning", "id": "r-paired"},
                {"type": "function_call", "call_id": "paired"},
                {"type": "function_call_output", "call_id": "paired", "output": "ok"},
                {"type": "message", "id": "m-final"},
            ]
            compacted = compactor.snip_compact(messages, max_messages=5)
            self.assertTrue(any(
                compactor.item_value(item, "id") == "r-paired"
                for item in compacted
            ))

            transcript = compactor.write_transcript([FakeSdkItem()])
            structured = json.loads(transcript.read_text(encoding="utf-8"))
            self.assertEqual(structured["type"], "reasoning")
            self.assertEqual(structured["mode"], "json")

        for marker in s08.CONTEXT_ERROR_MARKERS:
            self.assertTrue(s08.is_context_too_long_error(RuntimeError(marker)))
        self.assertFalse(s08.is_context_too_long_error(RuntimeError("network timeout")))


if __name__ == "__main__":
    unittest.main()
