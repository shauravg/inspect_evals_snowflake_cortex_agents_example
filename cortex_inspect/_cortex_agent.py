import json
import os
import re
import uuid
from typing import Any

import aiohttp

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._model_output import ChatCompletionChoice
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

# XML tag used to bracket tool calls in the model's text output
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_TOOL_SYSTEM_PREFIX = """\
You have access to the following tools. To call a tool, output a tool call block \
anywhere in your response using this exact format:

<tool_call>{"tool": "<tool_name>", "input": <json_object>}</tool_call>

Available tools:
"""


class CortexAgentModelAPI(ModelAPI):
    """Wraps the Snowflake Cortex Agents :run endpoint as an inspect-ai model.

    Model name format:
        cortex-agents/<database>/<schema>/<agent_name>

    Required environment variables:
        SNOWFLAKE_CORTEX_TOKEN    — Snowflake PAT or JWT bearer token
        SNOWFLAKE_CORTEX_BASE_URL — https://<account>.snowflakecomputing.com

    Optional environment variable:
        CORTEX_INSPECT_DEBUG=1    — print raw response body to stderr

    Both credentials can also be passed via -M flags:
        inspect eval my_task.py \\
            --model cortex-agents/mydb/myschema/my_agent \\
            -M api_key=<token> \\
            -M base_url=https://<account>.snowflakecomputing.com
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars or ["SNOWFLAKE_CORTEX_TOKEN"],
            config=config,
        )

        self._token = api_key or os.environ.get("SNOWFLAKE_CORTEX_TOKEN", "")
        self._base_url = (
            base_url or os.environ.get("SNOWFLAKE_CORTEX_BASE_URL", "")
        ).rstrip("/")
        self._debug = os.environ.get("CORTEX_INSPECT_DEBUG", "") == "1"

        if not self._token:
            raise ValueError(
                "Snowflake bearer token is required. Set SNOWFLAKE_CORTEX_TOKEN "
                "or pass -M api_key=<token>."
            )
        if not self._base_url:
            raise ValueError(
                "Snowflake base URL is required. Set SNOWFLAKE_CORTEX_BASE_URL "
                "or pass -M base_url=https://<account>.snowflakecomputing.com."
            )

        # inspect-ai strips the provider prefix before calling __init__, so
        # model_name arrives as "mydb/myschema/my_agent" (not "cortex-agents/...")
        segments = model_name.split("/")
        if len(segments) != 3:
            raise ValueError(
                f"Invalid model name '{model_name}'. Expected format: "
                "cortex-agents/<database>/<schema>/<agent_name>"
            )
        self._db, self._schema, self._agent = segments

    # ------------------------------------------------------------------
    # Core: call the Cortex Agents :run endpoint
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        url = (
            f"{self._base_url}/api/v2/databases/{self._db}"
            f"/schemas/{self._schema}/agents/{self._agent}:run"
        )

        # `submit` is an inspect-ai protocol pseudo-tool, not a real Snowflake
        # tool.  If we pass it to the Cortex Agent it will try to execute it as
        # a server-side tool and fail.  Instead, strip it out and auto-wrap the
        # agent's text response as a submit() call after the fact.
        submit_tool = next((t for t in tools if t.name == "submit"), None)
        real_tools = [t for t in tools if t.name != "submit"]

        # Build messages — only inject tool descriptions for real (non-submit) tools
        messages = _build_messages(input, real_tools)

        request_body: dict[str, Any] = {
            "messages": messages,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)  # 10 min — agents can be slow
        ) as session:
            async with session.post(url, json=request_body, headers=headers) as resp:
                resp.raise_for_status()
                status = resp.status
                body_bytes = await resp.read()

        body_text = body_bytes.decode("utf-8")
        if self._debug:
            import sys
            print(f"[cortex-agents] raw response:\n{body_text}", file=sys.stderr)

        response_json: dict[str, Any] = json.loads(body_text)
        raw_response = {"status": status, "body": response_json}

        full_text = _extract_text(response_json)

        if submit_tool:
            # Auto-submit: wrap the agent's text answer as a submit() tool call.
            # The agent answered naturally; we handle the protocol on its behalf.
            answer_param = _first_param_name(submit_tool) or "answer"
            tool_calls = [
                ToolCall(
                    id=str(uuid.uuid4())[:8],
                    function="submit",
                    arguments={answer_param: full_text},
                    type="function",
                )
            ]
            message = ChatMessageAssistant(
                content=full_text,
                tool_calls=tool_calls,
                source="generate",
            )
            output = ModelOutput.from_message(message, stop_reason="tool_calls")
        else:
            # No submit tool — check for any manually emitted <tool_call> blocks
            parsed_tool_calls = _parse_tool_calls(full_text)
            if parsed_tool_calls:
                visible_text = _TOOL_CALL_RE.sub("", full_text).strip()
                message = ChatMessageAssistant(
                    content=visible_text,
                    tool_calls=parsed_tool_calls,
                    source="generate",
                )
                output = ModelOutput.from_message(message, stop_reason="tool_calls")
            else:
                output = ModelOutput.from_content(
                    model=self.model_name,
                    content=full_text,
                )

        call = ModelCall.create(
            request={"url": url, "body": request_body},
            response=raw_response,
        )
        return output, call

    # ------------------------------------------------------------------
    # Tuning
    # ------------------------------------------------------------------

    def max_connections(self) -> int:
        return 4

    def should_retry(self, ex: Exception) -> bool:
        if isinstance(ex, aiohttp.ServerTimeoutError):
            return False  # timed out once — won't help to retry
        if isinstance(ex, aiohttp.ClientResponseError):
            return ex.status in (429, 500, 502, 503, 504)
        return False

    def connection_key(self) -> str:
        return self._base_url


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_messages(
    input: list[ChatMessage],
    tools: list[ToolInfo],
) -> list[dict[str, Any]]:
    """Convert inspect-ai ChatMessages → Cortex Agents message dicts.

    Cortex Agents :run only accepts role "user" and "assistant".
    System messages are merged into the first user message (prepended).
    Tool descriptions are also folded into that same first user message.
    """
    tool_block = _build_tool_block(tools) if tools else ""

    # Collect all system messages into a single prefix block
    system_parts: list[str] = []
    non_system: list[ChatMessage] = []
    for msg in input:
        if msg.role == "system":
            text = msg.text if hasattr(msg, "text") and msg.text else str(msg.content)
            system_parts.append(text)
        else:
            non_system.append(msg)

    # Build the prefix: tool instructions first, then system messages
    prefix_parts = ([tool_block] if tool_block else []) + system_parts
    prefix = "\n\n".join(prefix_parts)

    messages: list[dict[str, Any]] = []
    prefix_injected = False

    for msg in non_system:
        content = msg.text if hasattr(msg, "text") and msg.text else str(msg.content)

        # Prepend into the first user message
        if not prefix_injected and prefix and msg.role == "user":
            content = prefix + "\n\n" + content
            prefix_injected = True

        messages.append({
            "role": msg.role,
            "content": [{"type": "text", "text": content}],
        })

    # No user message existed — create one to carry the prefix
    if not prefix_injected and prefix:
        messages.insert(0, {
            "role": "user",
            "content": [{"type": "text", "text": prefix}],
        })

    return messages


def _build_tool_block(tools: list[ToolInfo]) -> str:
    """Render tool descriptions as a system-prompt block."""
    lines = [_TOOL_SYSTEM_PREFIX]
    for tool in tools:
        schema = tool.parameters.model_dump() if hasattr(tool.parameters, "model_dump") else {}
        lines.append(
            f"- name: {tool.name}\n"
            f"  description: {tool.description}\n"
            f"  input_schema: {json.dumps(schema, indent=2)}\n"
        )
    lines.append(
        "\nWhen you have finished reasoning, output your final answer as plain text."
    )
    return "\n".join(lines)


def _first_param_name(tool: ToolInfo) -> str | None:
    """Return the name of the first parameter in a tool's input schema."""
    try:
        schema = tool.parameters.model_dump() if hasattr(tool.parameters, "model_dump") else {}
        props = schema.get("properties", {})
        return next(iter(props), None)
    except Exception:
        return None


def _parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Find <tool_call>...</tool_call> blocks in text and parse them."""
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return None

    tool_calls: list[ToolCall] = []
    for raw in matches:
        try:
            obj = json.loads(raw.strip())
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4())[:8],
                    function=obj["tool"],
                    arguments=obj.get("input", {}),
                    type="function",
                )
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return tool_calls if tool_calls else None


def _extract_text(response: dict[str, Any]) -> str:
    """Extract plain text blocks from a Cortex Agents non-streaming response.

    The content array can contain items with type "text", "thinking",
    "tool_use", "tool_result", "table", "chart", etc. For eval purposes
    we concatenate all top-level text blocks.
    """
    parts: list[str] = []
    for item in response.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts).strip()
