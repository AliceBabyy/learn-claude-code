#!/usr/bin/env python3
"""
s12_cron_scheduler.py - Cron Scheduler

    +--------------------------+   09:00   +-----------------------+
    | 0 9 * * *               | --------> | [Scheduled] run tests |
    | prompt: "run tests"      |           +-----------+-----------+
    +--------------------------+                       |
          scheduled_jobs                    cron_queue | agent idle
                                                        v
                                                +-------------+
                                                | Agent Loop  |
                                                +-------------+
"""

import glob
import json
import os
import secrets
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
MODEL = os.getenv("OPENAI_MODEL_ID")
if not MODEL:
    raise RuntimeError("缺少 OPENAI_MODEL_ID，请在项目根目录的 .env 中配置模型名称")

SYSTEM = (
    f"你是位于 {WORKDIR} 的编程智能体。使用工具解决任务。"
    "需要在未来本地时间启动的工作使用 schedule_cron。"
)


# -- From s04: tool implementations --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"错误：命令以状态码 {result.returncode} 退出\n{output}"
        return output[:50000] if output else "（没有输出）"
    except subprocess.TimeoutExpired:
        return "错误：执行超时（120 秒）"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"错误：{error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"已向 {path} 写入 {len(content)} 字节"
    except Exception as error:
        return f"错误：{error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text()
        if old_text not in text:
            return f"错误：在 {path} 中未找到指定文本"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"已编辑 {path}"
    except Exception as error:
        return f"错误：{error}"


def run_glob(pattern: str) -> str:
    try:
        matches = [
            match
            for match in glob.glob(pattern, root_dir=WORKDIR)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        ]
        return "\n".join(matches) if matches else "（没有匹配项）"
    except Exception as error:
        return f"错误：{error}"


TOOLS = [
    {"type": "function", "name": "bash", "description": "执行一条 Shell 命令。",
     "parameters": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"], "additionalProperties": False}},
    {"type": "function", "name": "read_file", "description": "读取文件内容。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"], "additionalProperties": False}},
    {"type": "function", "name": "write_file", "description": "将内容写入文件。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"], "additionalProperties": False}},
    {"type": "function", "name": "edit_file", "description": "精确替换文件中首次出现的指定文本。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"], "additionalProperties": False}},
    {"type": "function", "name": "glob", "description": "查找与 glob 模式匹配的文件。",
     "parameters": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"], "additionalProperties": False}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- From s04: hooks and permission checks --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def request_permission(tool_name: str, arguments: dict, reason: str) -> str | None:
    if threading.current_thread() is not threading.main_thread():
        return "权限拒绝：定时任务轮次不能请求交互式批准"

    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   工具：{tool_name}({arguments})")
    choice = input("   Allow? [y/N] ").strip().lower()
    if choice not in ("y", "yes"):
        return "权限拒绝：用户未授权"
    return None


def permission_hook(tool_name: str, arguments: dict):
    if tool_name == "bash":
        command = arguments.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "权限拒绝：命令命中禁止列表"
        if any(keyword in command for keyword in DESTRUCTIVE):
            return request_permission(tool_name, arguments, "可能具有破坏性的命令")

    if tool_name in ("read_file", "write_file", "edit_file"):
        path = arguments.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            return request_permission(tool_name, arguments, "正在访问工作区外部")
    return None


def log_hook(tool_name: str, arguments: dict):
    preview = str(list(arguments.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({preview})\033[0m")
    return None


def large_output_hook(tool_name: str, arguments: dict, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] {tool_name} 输出过大："
            f"{len(str(output))} chars\033[0m"
        )
    return None


def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit：当前工作目录为 {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1 for item in messages
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "function_call"
    )
    print(f"\033[90m[HOOK] Stop：本次会话调用了 {tool_count} 次工具\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# -- New in s12: cron jobs --

@dataclass
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool
    pending_delivery: bool = False
    last_fired: str | None = None


scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.RLock()


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)
    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, day, month, weekday = fields
    cron_weekday = (moment.weekday() + 1) % 7
    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False

    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches


def _validate_cron_field(field: str, minimum: int, maximum: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), minimum, maximum)
            if error:
                return error
        return None
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        start_value, end_value = int(start), int(end)
        if start_value > end_value:
            return f"Range start is greater than end: {field}"
        if start_value < minimum or end_value > maximum:
            return f"Range {field} is outside [{minimum}-{maximum}]"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < minimum or value > maximum:
        return f"Value {value} is outside [{minimum}-{maximum}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    field_rules = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]
    for field, (name, minimum, maximum) in zip(fields, field_rules):
        error = _validate_cron_field(field, minimum, maximum)
        if error:
            return f"{name}: {error}"
    return None


def save_durable_jobs():
    with cron_lock:
        payload = [
            asdict(job)
            for job in scheduled_jobs.values()
            if job.durable
        ]
        temporary = DURABLE_PATH.with_name(
            f"{DURABLE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2))
            os.replace(temporary, DURABLE_PATH)
        finally:
            temporary.unlink(missing_ok=True)


def load_durable_jobs():
    if not DURABLE_PATH.exists():
        return
    try:
        payload = json.loads(DURABLE_PATH.read_text())
        if not isinstance(payload, list):
            raise ValueError("expected a JSON list")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"  [cron] could not load {DURABLE_PATH.name}: {error}")
        return

    loaded = 0
    with cron_lock:
        for item in payload:
            try:
                job = CronJob(**item)
                error = validate_cron(job.cron)
                if error:
                    raise ValueError(error)
                if not job.id.startswith("cron_"):
                    raise ValueError("invalid job ID")
                if not job.prompt.strip():
                    raise ValueError("prompt cannot be empty")
            except (TypeError, ValueError) as error:
                print(f"  [cron] skipped invalid saved job: {error}")
                continue
            scheduled_jobs[job.id] = job
            if job.pending_delivery:
                cron_queue.append(job)
            loaded += 1
    if loaded:
        print(f"  [cron] loaded {loaded} durable job(s)")


def new_cron_id() -> str:
    for _ in range(100):
        job_id = f"cron_{secrets.token_hex(4)}"
        if job_id not in scheduled_jobs:
            return job_id
    raise RuntimeError("Could not allocate a cron job ID")


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    error = validate_cron(cron)
    if error:
        return error
    if not prompt.strip():
        return "Prompt cannot be empty"

    with cron_lock:
        job = CronJob(
            id=new_cron_id(),
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        scheduled_jobs[job.id] = job
        try:
            if durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs.pop(job.id, None)
            raise
    print(f"  [cron] scheduled {job.id}: {cron} -> {prompt[:60]}")
    return job


def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.get(job_id)
        if job is None:
            return f"Job {job_id} not found"

        previous_queue = list(cron_queue)
        scheduled_jobs.pop(job_id)
        cron_queue[:] = [queued for queued in cron_queue if queued.id != job_id]
        try:
            if job.durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs[job_id] = job
            cron_queue[:] = previous_queue
            raise
    print(f"  [cron] cancelled {job_id}")
    return f"Cancelled {job_id}"


def _enqueue_due_job(job: CronJob, minute_marker: str | None = None):
    old_pending = job.pending_delivery
    old_last_fired = job.last_fired
    job.pending_delivery = True
    if minute_marker is not None:
        job.last_fired = minute_marker
    try:
        if job.durable:
            save_durable_jobs()
    except Exception:
        job.pending_delivery = old_pending
        job.last_fired = old_last_fired
        raise
    cron_queue.append(job)


def poll_due_jobs(moment: datetime):
    minute_marker = moment.strftime("%Y-%m-%d %H:%M")
    with cron_lock:
        for job in list(scheduled_jobs.values()):
            try:
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue
                if cron_matches(job.cron, moment):
                    _enqueue_due_job(job, minute_marker)
                    print(f"  [cron] due {job.id}: {job.prompt[:60]}")
            except Exception as error:
                print(f"  [cron] could not enqueue {job.id}: {error}")


def consume_cron_queue() -> list[CronJob]:
    with cron_lock:
        jobs = list(cron_queue)
        cron_queue.clear()
    return jobs


def acknowledge_cron_jobs(jobs: list[CronJob]):
    changed: list[tuple[CronJob, bool]] = []
    removed: list[CronJob] = []
    with cron_lock:
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                continue
            changed.append((current, current.pending_delivery))
            if current.recurring:
                current.pending_delivery = False
            else:
                removed.append(current)
                scheduled_jobs.pop(current.id)

        try:
            if any(job.durable for job, _ in changed):
                save_durable_jobs()
        except Exception:
            for job in removed:
                scheduled_jobs[job.id] = job
            for job, pending in changed:
                job.pending_delivery = pending
            queued_ids = {job.id for job in cron_queue}
            for job, _ in changed:
                if job.id not in queued_ids:
                    cron_queue.append(job)
            raise


def restore_cron_jobs(jobs: list[CronJob]):
    with cron_lock:
        queued_ids = {job.id for job in cron_queue}
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                continue
            current.pending_delivery = True
            if current.id not in queued_ids:
                cron_queue.append(current)
                queued_ids.add(current.id)


def has_cron_queue() -> bool:
    with cron_lock:
        return bool(cron_queue)


def run_schedule_cron(cron: str, prompt: str, recurring: bool = True,
                      durable: bool = True) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: {cron} -> {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."

    lines = []
    for job in jobs:
        frequency = "recurring" if job.recurring else "one-shot"
        storage = "durable" if job.durable else "session"
        lines.append(
            f"{job.id}: {job.cron} -> {job.prompt[:60]} "
            f"[{frequency}, {storage}]"
        )
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


TOOLS.extend([
    {"type": "function", "name": "schedule_cron",
     "description": "使用5字段Cron表达式安排提示词任务。",
     "parameters": {"type": "object",
                      "properties": {
                          "cron": {"type": "string"},
                          "prompt": {"type": "string"},
                          "recurring": {"type": "boolean"},
                          "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"], "additionalProperties": False}},
    {"type": "function", "name": "list_crons", "description": "列出已安排的Cron任务。",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "cancel_cron", "description": "按ID取消Cron任务。",
     "parameters": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"], "additionalProperties": False}},
])

TOOL_HANDLERS.update({
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
})


def execute_tool(tool_name: str, arguments: dict) -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, arguments)
    if blocked is not None:
        return str(blocked)

    handler = TOOL_HANDLERS.get(tool_name)
    try:
        output = handler(**arguments) if handler else f"错误：未知工具 {tool_name}"
    except Exception as error:
        output = f"错误：{error}"
    trigger_hooks("PostToolUse", tool_name, arguments, output)
    return str(output)


# -- Scheduler and agent loop --

RUNTIME_STOP = threading.Event()
runtime_threads: list[threading.Thread] = []
runtime_started = False
runtime_lock = threading.Lock()
agent_lock = threading.Lock()
session_history: list = []


def cron_scheduler_loop(stop_event: threading.Event = RUNTIME_STOP):
    while not stop_event.wait(1.0):
        poll_due_jobs(datetime.now())


def agent_loop(messages: list, context: dict | None = None):
    fired = consume_cron_queue()
    scheduled_start = len(messages)
    for job in fired:
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
        print(f"  [cron] delivered {job.id}: {job.prompt[:60]}")

    waiting_for_ack = list(fired)
    while True:
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM,
                input=messages,
                tools=TOOLS,
                max_output_tokens=8000,
            )
        except Exception as error:
            if waiting_for_ack:
                del messages[scheduled_start:]
                restore_cron_jobs(waiting_for_ack)
            print(f"  [error] {type(error).__name__}: {error}")
            return context

        messages.extend(response.output)
        if waiting_for_ack:
            try:
                acknowledge_cron_jobs(waiting_for_ack)
            except Exception as error:
                print(f"  [cron] acknowledgement failed: {error}")
            waiting_for_ack = []

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return context

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)
            output = execute_tool(tool_call.name, arguments)
            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": output,
            })


def print_latest_assistant_text(messages: list):
    for item in reversed(messages):
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "output_text":
                print(part.get("text", "") if isinstance(part, dict) else part.text)
        return


def run_agent_turn_locked(user_query: str | None = None):
    if user_query is not None:
        trigger_hooks("UserPromptSubmit", user_query)
        session_history.append({"role": "user", "content": user_query})
    agent_loop(session_history)
    print_latest_assistant_text(session_history)
    print()


def queue_processor_loop(stop_event: threading.Event = RUNTIME_STOP):
    while not stop_event.wait(0.2):
        if not has_cron_queue() or not agent_lock.acquire(blocking=False):
            continue
        try:
            if has_cron_queue():
                run_agent_turn_locked()
        finally:
            agent_lock.release()


def start_runtime_threads():
    global runtime_started
    with runtime_lock:
        if runtime_started:
            return
        load_durable_jobs()
        RUNTIME_STOP.clear()
        runtime_threads.extend([
            threading.Thread(
                target=cron_scheduler_loop,
                name="cron-scheduler",
                daemon=True,
            ),
            threading.Thread(
                target=queue_processor_loop,
                name="cron-queue-processor",
                daemon=True,
            ),
        ])
        for thread in runtime_threads:
            thread.start()
        runtime_started = True


def stop_runtime_threads():
    global runtime_started
    with runtime_lock:
        if not runtime_started:
            return
        RUNTIME_STOP.set()
        for thread in runtime_threads:
            thread.join(timeout=1)
        runtime_threads.clear()
        runtime_started = False


if __name__ == "__main__":
    print("s12：Cron调度器 - 按本地时间运行提示词任务")
    print("输入问题后按回车发送，输入 q 或 exit 退出。\n")
    start_runtime_threads()
    try:
        while True:
            try:
                query = input("\033[36ms12 >> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            with agent_lock:
                run_agent_turn_locked(query)
    finally:
        stop_runtime_threads()
