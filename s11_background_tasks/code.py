#!/usr/bin/env python3
"""
s11_background_tasks.py - Background Tasks

    Main thread                              Background thread
    +------------------------------+         +----------------------+
    | bash(run_in_background=True) | ------> | run command          |
    | return bg_id                 |         | queue result         |
    | continue agent loop          | <------ +----------------------+
    | next turn: collect           |
    +------------------------------+
"""

import atexit
import glob
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
MODEL = os.getenv("OPENAI_MODEL_ID")
if not MODEL:
    raise RuntimeError("缺少 OPENAI_MODEL_ID，请在项目根目录的 .env 中配置模型名称")

SYSTEM = (
    f"你是位于 {WORKDIR} 的编程智能体。使用工具解决任务。"
    "只有独立的 Bash 命令才设置 run_in_background=true。"
)


# -- From s04: tool implementations --

_shell_processes: set[subprocess.Popen] = set()
_shell_process_lock = threading.RLock()


def _stop_process_group(process: subprocess.Popen):
    """Stop processes that remain in the command's original process group."""
    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGKILL"):
        signals.append(signal.SIGKILL)
    for sig in signals:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.terminate()
        except (ProcessLookupError, OSError):
            return
        time.sleep(0.05)


def _stop_all_shell_processes():
    with _shell_process_lock:
        processes = list(_shell_processes)
    for process in processes:
        _stop_process_group(process)


def _handle_termination_signal(signum, _frame):
    _stop_all_shell_processes()
    raise SystemExit(128 + signum)


atexit.register(_stop_all_shell_processes)
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=True,
        )
        with _shell_process_lock:
            _shell_processes.add(process)
        stdout, stderr = process.communicate(timeout=120)
        output = (stdout + stderr).strip()
        return (output[:50000] if output else "（没有输出）"), process.returncode
    except subprocess.TimeoutExpired:
        return "错误：执行超时（120 秒）", None
    except OSError as error:
        return f"错误：{type(error).__name__}: {error}", None
    finally:
        if process is not None:
            _stop_process_group(process)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            with _shell_process_lock:
                _shell_processes.discard(process)


def _format_bash_result(output: str, exit_code: int | None) -> str:
    if exit_code in (0, None):
        return output
    return f"错误：命令以状态码 {exit_code} 退出\n{output}"


def run_bash(command: str, run_in_background: bool = False) -> str:
    return _format_bash_result(*_run_bash_process(command))


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...（还有 {len(lines) - limit} 行）"]
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
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
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


def permission_hook(tool_name: str, arguments: dict):
    if tool_name == "bash":
        command = arguments.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "权限拒绝：命令命中禁止列表"
        if any(keyword in command for keyword in DESTRUCTIVE):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   工具：{tool_name}({arguments})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "权限拒绝：用户未授权"

    if tool_name in ("read_file", "write_file", "edit_file"):
        path = arguments.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   工具：{tool_name}({arguments})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "权限拒绝：用户未授权"
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
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1 for item in messages
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "function_call"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def call_tool(tool_name: str, arguments: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    try:
        output = handler(**arguments) if handler else f"错误：未知工具 {tool_name}"
    except Exception as error:
        output = f"错误：{error}"
    return str(output)


# -- New in s11: background execution --

class BackgroundManager:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, tool_name: str, arguments: dict, call_id: str) -> str:
        if tool_name != "bash":
            raise ValueError("只有 Bash 命令可以在后台运行")
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash 命令不能为空")

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self.tasks[task_id] = {
                "call_id": call_id,
                "command": command,
                "status": "running",
            }

        thread = threading.Thread(
            target=self._run,
            args=(task_id, command),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        try:
            output, exit_code = _run_bash_process(command)
            result = _format_bash_result(output, exit_code)
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            result = f"错误：{type(error).__name__}: {error}"
            status = "failed"

        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            self._ready.append(task_id)

    def collect(self) -> list[str]:
        with self._lock:
            ready = []
            for task_id in self._ready:
                task = self.tasks.pop(task_id, None)
                result = self.results.pop(task_id, "")
                if task is not None:
                    ready.append((task_id, task, result))
            self._ready.clear()

        notifications = []
        for task_id, task, result in ready:
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


BACKGROUND = BackgroundManager()
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash"
        and tool_input.get("run_in_background") is True
    )


def start_background_task(tool_name: str, arguments: dict, call_id: str) -> str:
    return BACKGROUND.start(tool_name, arguments, call_id)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()


def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0

    blocks = "\n\n".join(notifications)
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        messages[-1]["content"] = f"{content}\n\n{blocks}" if content else blocks
    else:
        messages.append({"role": "user", "content": blocks})
    return len(notifications)


def execute_tool(tool_name: str, arguments: dict, call_id: str) -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, arguments)
    if blocked is not None:
        return str(blocked)

    if should_run_background(tool_name, arguments):
        try:
            task_id = start_background_task(tool_name, arguments, call_id)
            output = (
                f"[后台任务 {task_id} 已启动] "
                "结果将在后续轮次收集。"
            )
        except Exception as error:
            output = f"错误：{error}"
    else:
        output = call_tool(tool_name, arguments)

    trigger_hooks("PostToolUse", tool_name, arguments, output)
    return output


# -- Agent loop --

def agent_loop(messages: list):
    while True:
        inject_background_results(messages)
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,
            input=messages,
            tools=TOOLS,
            max_output_tokens=8000,
        )
        messages.extend(response.output)

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return response.output_text

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)
            output = execute_tool(tool_call.name, arguments, tool_call.call_id)
            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": output,
            })


if __name__ == "__main__":
    print("s11：后台任务 - 显式后台执行 Bash 命令")
    print("输入问题后按回车发送，输入 q 或 exit 退出。\n")
    print(f"请求地址：{client.base_url}responses")

    history = []
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        final_text = agent_loop(history)
        if final_text:
            print(final_text)
        print()
