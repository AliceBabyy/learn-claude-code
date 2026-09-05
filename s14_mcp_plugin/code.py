#!/usr/bin/env python3
"""
s14：MCP 工具 - 发现外部工具并将其加入智能体循环。

运行：python s14_mcp_plugin/code.py
依赖：pip install openai python-dotenv，并在 .env 中配置 OPENAI_API_KEY

    connect_mcp("docs")
              |
              v
    +------------------+     tools/list     +------------------+
    | 智能体框架       | <----------------- | MCP 服务端       |
    |                  |                    | docs             |
    | 内置工具         |     tools/call     |                  |
    | + MCP 工具       | -----------------> | search           |
    +--------+---------+                    | get_version      |
             |                              +------------------+
             v
    +-----------------------------------------------+
    | bash | read | write | edit | glob | connect  |
    | mcp__docs__search | mcp__docs__get_version   |
    +-----------------------------------------------+
"""

import glob
import json
import os
import re
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind("set bind-tty-special-chars off")
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
MODEL = os.getenv("OPENAI_MODEL_ID")
if not MODEL:
    raise RuntimeError("缺少 OPENAI_MODEL_ID，请在项目根目录的 .env 中配置模型名称")

BASE_SYSTEM = (
    f"你是位于 {WORKDIR} 的编程智能体。使用内置工具和已连接的 MCP 工具"
    "解决任务。使用服务端之前先调用 connect_mcp。"
)


# -- 来自 s04：基础工具 --

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
        output = output[:50000] if output else "（没有输出）"
        if result.returncode:
            return f"错误：命令退出状态码为 {result.returncode}\n{output}"
        return output
    except subprocess.TimeoutExpired:
        return "错误：执行超时（120 秒）"
    except OSError as exc:
        return f"错误：{type(exc).__name__}：{exc}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...（还有 {len(lines) - limit} 行）"]
        return "\n".join(lines)
    except Exception as exc:
        return f"错误：{exc}"


def run_write(path: str, content: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已向 {path} 写入 {len(content)} 字节"
    except Exception as exc:
        return f"错误：{exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        content = target.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            return f"错误：预期文本出现 1 次，实际找到 {count} 次"
        target.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return f"已编辑 {path}"
    except Exception as exc:
        return f"错误：{exc}"


def run_glob(pattern: str) -> str:
    try:
        matches = [
            match
            for match in glob.glob(pattern, root_dir=WORKDIR)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR.resolve())
        ]
        return "\n".join(matches[:200]) if matches else "（没有匹配项）"
    except Exception as exc:
        return f"错误：{exc}"


BASE_TOOLS = [
    {"type": "function", "name": "bash", "description": "执行一条 Shell 命令。",
     "parameters": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"type": "function", "name": "read_file", "description": "读取文件内容。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"type": "function", "name": "write_file", "description": "将内容写入文件。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"type": "function", "name": "edit_file", "description": "精确替换文本一次。",
     "parameters": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"type": "function", "name": "glob", "description": "按 glob 模式查找文件。",
     "parameters": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

BASE_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- s14 新增：MCP 发现与分发 --

class MCPClient:
    """用于模拟 MCP tools/list 和 tools/call 的小型进程内客户端。"""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register(self, tool_defs: list[dict], handlers: dict[str, callable]):
        names = [tool.get("name") for tool in tool_defs]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("每个 MCP 工具都必须有非空名称")
        if len(set(names)) != len(names):
            raise ValueError(f"MCP 服务端 {self.name!r} 存在重复工具名称")
        missing = [name for name in names if name not in handlers]
        if missing:
            raise ValueError(f"缺少 MCP 处理函数：{', '.join(missing)}")
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name: str, args: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP 错误：未知工具“{tool_name}”"
        try:
            return str(handler(**args))
        except Exception as exc:
            return f"MCP 错误：{type(exc).__name__}：{exc}"


mcp_clients: dict[str, MCPClient] = {}
mcp_tool_policies: dict[str, str] = {}
_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# 授权策略来自主机配置，绝不读取服务端描述来决定。
MCP_HOST_POLICY = {
    ("docs", "search"): "allow",
    ("docs", "get_version"): "allow",
    ("deploy", "status"): "allow",
    ("deploy", "trigger"): "confirm",
}


def normalize_mcp_name(name: str) -> str:
    """替换模型工具名称允许字符范围之外的字符。"""
    normalized = _DISALLOWED_CHARS.sub("_", name)
    if not normalized:
        raise ValueError("MCP 名称规范化后不能为空字符串")
    return normalized


def _mock_server_docs() -> MCPClient:
    server = MCPClient("docs")
    server.register(
        tool_defs=[
            {
                "name": "search",
                "description": "搜索文档。",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "get_version",
                "description": "获取文档 API 版本。",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "search": lambda query: f"[文档] 找到 3 条与“{query}”相关的结果",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return server


def _mock_server_deploy() -> MCPClient:
    server = MCPClient("deploy")
    server.register(
        tool_defs=[
            {
                "name": "trigger",
                "description": "触发一次部署。",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
                "annotations": {"destructiveHint": True},
            },
            {
                "name": "status",
                "description": "检查部署状态。",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "trigger": lambda service: f"[部署] 已触发：{service}",
            "status": lambda service: f"[部署] {service}：运行中（v1.4.2）",
        },
    )
    return server


MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}


def connect_mcp(name: str) -> str:
    if name in mcp_clients:
        return f"MCP 服务端“{name}”已连接"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        return f"未知服务端“{name}”。可用服务端：{', '.join(MOCK_SERVERS)}"
    server = factory()
    mcp_clients[name] = server
    names = ", ".join(tool["name"] for tool in server.tools)
    print(f"  [MCP] 已连接：{name} -> {names}")
    return (
        f"已连接 MCP 服务端“{name}”。"
        f"发现 {len(server.tools)} 个工具：{names}"
    )


def run_connect_mcp(name: str) -> str:
    return connect_mcp(name)


CONNECT_TOOL = {
    "type": "function",
    "name": "connect_mcp",
    "description": "连接 MCP 服务端并发现其工具。",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["docs", "deploy"]}},
        "required": ["name"],
    },
}

BUILTIN_TOOLS = [*BASE_TOOLS, CONNECT_TOOL]
BUILTIN_HANDLERS = {**BASE_HANDLERS, "connect_mcp": run_connect_mcp}


def assemble_tool_pool() -> tuple[list[dict], dict[str, callable]]:
    """将内置工具与所有已连接服务端的工具合并。"""
    global mcp_tool_policies
    tools = list(BUILTIN_TOOLS)
    handlers = dict(BUILTIN_HANDLERS)
    policies: dict[str, str] = {}
    origins = {
        tool["name"]: f"内置工具 {tool['name']!r}"
        for tool in tools
    }

    for server_name, server in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for tool_def in server.tools:
            raw_name = tool_def["name"]
            safe_tool = normalize_mcp_name(raw_name)
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            if len(prefixed) > 64:
                raise ValueError(f"MCP 工具名称超过 64 个字符：{prefixed}")
            origin = f"MCP 工具 {server_name!r}/{raw_name!r}"
            if prefixed in origins:
                raise ValueError(
                    "MCP 工具名称规范化后发生冲突："
                    f"{prefixed!r} 同时映射到 {origins[prefixed]} 和 {origin}"
                )
            schema = tool_def.get("inputSchema", {})
            if not isinstance(schema, dict) or schema.get("type", "object") != "object":
                raise ValueError(f"{origin} 的输入 schema 无效")
            origins[prefixed] = origin
            tools.append({
                "type": "function",
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "parameters": schema,
            })
            handlers[prefixed] = (
                lambda *, client=server, tool=raw_name, **kwargs:
                client.call_tool(tool, kwargs)
            )
            policies[prefixed] = MCP_HOST_POLICY.get(
                (server_name, raw_name), "confirm"
            )

    mcp_tool_policies = policies
    return tools, handlers


def assemble_system_prompt() -> str:
    if not mcp_clients:
        return BASE_SYSTEM
    return BASE_SYSTEM + "\n\n已连接的 MCP 服务端：" + ", ".join(mcp_clients)


# -- 来自 s04：钩子和权限检查 --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def permission_hook(tool_name: str, arguments: dict):
    if tool_name == "bash":
        command = arguments.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"权限被拒绝：命令命中禁止列表 {pattern}"
        if any(keyword in command for keyword in DESTRUCTIVE):
            print(f"\n[权限] {tool_name}({arguments})")
            if input("是否允许？[y/N] ").strip().lower() not in {"y", "yes"}:
                return "权限被拒绝：用户未允许该操作"

    if tool_name in {"read_file", "write_file", "edit_file"}:
        raw_path = arguments.get("path", "")
        if not (WORKDIR / raw_path).resolve().is_relative_to(WORKDIR.resolve()):
            print(f"\n[权限] {tool_name}({arguments})")
            if input("是否允许？[y/N] ").strip().lower() not in {"y", "yes"}:
                return "权限被拒绝：用户未允许该操作"

    if tool_name.startswith("mcp__"):
        policy = mcp_tool_policies.get(tool_name, "confirm")
        if policy != "allow":
            print(f"\n[权限] 外部工具 {tool_name}({arguments})")
            if input("是否允许？[y/N] ").strip().lower() not in {"y", "yes"}:
                return "权限被拒绝：用户未允许该操作"
    return None


def log_hook(tool_name: str, arguments: dict):
    preview = str(list(arguments.values())[:2])[:60]
    print(f"[钩子] {tool_name}({preview})")
    return None


def large_output_hook(tool_name: str, arguments: dict, output):
    if len(str(output)) > 100000:
        print(f"[钩子] {tool_name} 输出过大：{len(str(output))} 个字符")
    return None


def context_hook(query: str):
    print(f"[钩子] 用户提交提示：当前工作目录为 {WORKDIR}")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1
        for item in messages
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
    )
    print(f"[钩子] 停止：本次会话使用了 {tool_count} 次工具调用")
    return None


register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(
    tool_name: str, arguments: dict, handlers: dict[str, callable]
) -> str:
    blocked = trigger_hooks("PreToolUse", tool_name, arguments)
    if blocked:
        return str(blocked)
    handler = handlers.get(tool_name)
    if not handler:
        return f"错误：未知工具 {tool_name}"
    try:
        output = str(handler(**arguments))
    except Exception as exc:
        output = f"错误：{type(exc).__name__}：{exc}"
    trigger_hooks("PostToolUse", tool_name, arguments, output)
    return output


# -- 使用动态工具池的智能体循环 --

def agent_loop(messages: list):
    round_number = 0
    while True:
        try:
            tools, handlers = assemble_tool_pool()
            round_number += 1
            tool_names = ", ".join(tool["name"] for tool in tools)
            print(f"当前轮次：{round_number}")
            print(f"本轮工具池：{tool_names}")
            response = client.responses.create(
                model=MODEL,
                instructions=assemble_system_prompt(),
                input=messages,
                tools=tools,
                max_output_tokens=8000,
            )
        except Exception as exc:
            messages.append({
                "role": "assistant",
                "content": f"[错误] {type(exc).__name__}：{exc}",
            })
            trigger_hooks("Stop", messages)
            return f"[错误] {type(exc).__name__}：{exc}"

        # 保留消息、推理项和函数调用等全部输出，供下一轮完整回传。
        messages.extend(response.output)
        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            trigger_hooks("Stop", messages)
            return response.output_text

        for tool_call in tool_calls:
            print(f"> {tool_call.name}")
            arguments = json.loads(tool_call.arguments)
            output = execute_tool(tool_call.name, arguments, handlers)
            print(output[:300])
            # 权限拒绝和 MCP 调用失败也必须回填，使模型获得失败原因。
            messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": output,
            })


if __name__ == "__main__":
    print("s14：MCP 工具")
    print("输入问题后按回车发送，输入 q 或 exit 退出。\n")
    print(f"请求地址：{client.base_url}responses")
    history = []

    while True:
        try:
            query = input("s14 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"q", "exit", ""}:
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        final_text = agent_loop(history)
        if final_text:
            print(final_text)
        print()

