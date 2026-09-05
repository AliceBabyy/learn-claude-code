import importlib.util
import os
import sys
import tempfile
import threading
import time
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "s12_cron_scheduler" / "code.py"


def load_lesson(workdir: Path):
    fake_openai = types.ModuleType("openai")
    fake_dotenv = types.ModuleType("dotenv")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = types.SimpleNamespace(create=None)

    fake_openai.OpenAI = FakeOpenAI
    fake_dotenv.load_dotenv = lambda override=True: None

    previous_modules = {
        "openai": sys.modules.get("openai"),
        "dotenv": sys.modules.get("dotenv"),
    }
    previous_cwd = Path.cwd()
    previous_model = os.environ.get("OPENAI_MODEL_ID")
    module_name = f"cron_scheduler_test_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, LESSON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    sys.modules["openai"] = fake_openai
    sys.modules["dotenv"] = fake_dotenv
    sys.modules[module_name] = module
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
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_s12_keeps_the_s04_kernel_and_adds_three_cron_tools():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))

        assert [tool["name"] for tool in lesson.TOOLS] == [
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "schedule_cron",
            "list_crons",
            "cancel_cron",
        ]
        assert all(tool["type"] == "function" and "parameters" in tool
                   for tool in lesson.TOOLS)
        assert set(lesson.HOOKS) == {
            "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"
        }
        assert not hasattr(lesson, "Task")
        assert not hasattr(lesson, "MEMORY_DIR")
        assert not hasattr(lesson, "background_tasks")


def test_import_does_not_start_runtime_threads():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))

        assert not lesson.runtime_started
        assert lesson.runtime_threads == []
        assert not any(
            thread.name in {"cron-scheduler", "cron-queue-processor"}
            for thread in threading.enumerate()
        )


def test_cron_validation_and_matching():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        monday_at_nine = datetime(2026, 8, 10, 9, 0)

        assert lesson.validate_cron("0 9 * * 1-5") is None
        assert lesson.cron_matches("0 9 * * 1-5", monday_at_nine)
        assert not lesson.cron_matches("30 9 * * 1-5", monday_at_nine)
        assert "hour" in lesson.validate_cron("0 24 * * *")
        assert "Expected 5 fields" in lesson.validate_cron("0 9 * *")


def test_schedule_retries_id_collisions_and_rolls_back_failed_persistence(
    monkeypatch: pytest.MonkeyPatch,
):
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        values = iter(["deadbeef", "deadbeef", "cafebabe", "bad0cafe"])
        monkeypatch.setattr(lesson.secrets, "token_hex", lambda _size: next(values))

        first = lesson.schedule_job("0 9 * * *", "first", durable=False)
        second = lesson.schedule_job("0 10 * * *", "second", durable=False)
        assert first.id == "cron_deadbeef"
        assert second.id == "cron_cafebabe"

        monkeypatch.setattr(
            lesson,
            "save_durable_jobs",
            lambda: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            lesson.schedule_job("0 11 * * *", "third", durable=True)
        assert "cron_bad0cafe" not in lesson.scheduled_jobs


def test_failed_model_call_restores_delivery_without_duplicate_message():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        job = lesson.CronJob(
            id="cron_retry",
            cron="* * * * *",
            prompt="retry the report",
            recurring=False,
            durable=True,
            pending_delivery=True,
        )
        lesson.scheduled_jobs[job.id] = job
        lesson.cron_queue.append(job)
        lesson.save_durable_jobs()
        lesson.client.responses.create = (
            lambda **_: (_ for _ in ()).throw(RuntimeError("offline"))
        )

        messages = []
        lesson.agent_loop(messages)

        assert messages == []
        assert [queued.id for queued in lesson.cron_queue] == [job.id]
        assert job.id in lesson.scheduled_jobs


def test_openai_history_keeps_reasoning_and_pairs_function_output():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        reasoning = types.SimpleNamespace(type="reasoning", id="rs_1")
        call = types.SimpleNamespace(
            type="function_call",
            call_id="call_list",
            name="list_crons",
            arguments="{}",
        )
        final = types.SimpleNamespace(type="message")
        responses = [
            types.SimpleNamespace(output=[reasoning, call], output_text=""),
            types.SimpleNamespace(output=[final], output_text="没有Cron任务。"),
        ]
        lesson.client.responses.create = lambda **_: responses.pop(0)
        messages = [{"role": "user", "content": "列出Cron任务"}]

        lesson.agent_loop(messages)

        assert messages[1] is reasoning
        assert messages[2] is call
        assert messages[3] == {
            "type": "function_call_output",
            "call_id": "call_list",
            "output": "No cron jobs.",
        }


def test_scheduled_turn_never_reads_interactive_permission_input():
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        results = []

        with patch("builtins.input", side_effect=AssertionError("input called")):
            thread = threading.Thread(
                target=lambda: results.append(
                    lesson.permission_hook("bash", {"command": "rm build.log"})
                )
            )
            thread.start()
            thread.join(timeout=1)

        assert results == ["权限拒绝：定时任务轮次不能请求交互式批准"]


def test_corrupt_durable_store_reports_an_error(capsys: pytest.CaptureFixture[str]):
    with tempfile.TemporaryDirectory() as tmp:
        lesson = load_lesson(Path(tmp))
        lesson.DURABLE_PATH.write_text("{broken")

        lesson.load_durable_jobs()

        assert "could not load .scheduled_tasks.json" in capsys.readouterr().out
        assert lesson.scheduled_jobs == {}


def test_s12_code_is_ascii():
    assert LESSON.read_text(encoding="utf-8")
