import asyncio
import base64
import json
import os
import uuid
from contextlib import suppress
from typing import Any, Optional

from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from smithy_http.aio.crt import AWSCRTHTTPClient

from mcp_client import MCPClient
from utils.prompt import SYSTEM_PROMPT


NOVA_INPUT_RATE = 16_000
NOVA_OUTPUT_RATE = 24_000


class NovaSonic:
    def __init__(
        self,
        mcp: MCPClient,
        model_id: str = "amazon.nova-2-sonic-v1:0",
    ):
        self.model_id = model_id
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.mcp = mcp

        self.client: Optional[AsyncBedrockRuntimeClient] = None
        self.stream = None
        self.response_task: Optional[asyncio.Task] = None
        self.is_active = False

        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self.audio_queue: asyncio.Queue = asyncio.Queue()

        self.twilio_resample_state = None

        self._audio_buffer = bytearray()
        self._audio_buffer_ms = 0
        self.CHUNK_MS = 100

        self._role: Optional[str] = None
        self._current_content_type: Optional[str] = None
        self._display_assistant_text = False

        self._in_tool_use = False
        self._tool_use_id: Optional[str] = None
        self._tool_name: Optional[str] = None
        self._tool_input: dict[str, Any] = {}

        self._call_sid = ""

        # Prevent caller audio from being sent before Twilio connects.
        self._ready_for_audio = asyncio.Event()

        # Barge-in state.
        self._nova_speaking = False
        self._generation_id = 0
        self._barge_in_fired = False
        self._barge_in_lock = asyncio.Lock()

        # main.py watches this event and sends Twilio "clear".
        self.barge_in_event = asyncio.Event()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _build_client(self) -> None:
        os.environ.setdefault("AWS_REGION", self.region)
        os.environ.setdefault("AWS_DEFAULT_REGION", self.region)

        config = await AsyncBedrockRuntimeConfig.resolve(
            region=self.region,
            transport=AWSCRTHTTPClient(),
        )

        self.client = AsyncBedrockRuntimeClient(config=config)

        print(
            "[nova] AsyncBedrockRuntimeClient created "
            f"with AWS CRT transport for region={self.region}"
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.stream is None:
            raise RuntimeError("Nova Sonic stream has not been initialized")

        raw = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(
                bytes_=raw.encode("utf-8")
            )
        )

        await self.stream.input_stream.send(chunk)

    def _clear_audio_queue_nowait(self) -> int:
        cleared = 0

        while True:
            try:
                self.audio_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break

        return cleared

    def _reset_tool_state(self) -> None:
        self._in_tool_use = False
        self._tool_use_id = None
        self._tool_name = None
        self._tool_input = {}

    # -------------------------------------------------------------------------
    # Barge-in handling
    # -------------------------------------------------------------------------

    async def _handle_barge_in(self, reason: str) -> None:
        async with self._barge_in_lock:
            if self._barge_in_fired:
                return

            self._barge_in_fired = True
            self._nova_speaking = False
            self._generation_id += 1

            cleared = self._clear_audio_queue_nowait()

            print(
                f"[nova] Barge-in reason={reason}, "
                f"generation={self._generation_id}, "
                f"cleared={cleared}"
            )

            # Signal main.py to clear Twilio's outbound playback buffer.
            self.barge_in_event.set()

    # -------------------------------------------------------------------------
    # Audio channel helpers
    # -------------------------------------------------------------------------

    async def _close_audio_channel(self) -> None:
        if self.stream is None:
            return

        try:
            await self._send(
                {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                        }
                    }
                }
            )

            print(
                "[nova] Audio channel closed "
                f"(name={self.audio_content_name})"
            )
        except Exception as exc:
            print(f"[nova] Audio channel close error: {exc}")

    async def _open_audio_channel(self) -> None:
        self.audio_content_name = str(uuid.uuid4())

        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_INPUT_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

        print(
            "[nova] Audio channel opened "
            f"(name={self.audio_content_name})"
        )

    async def _send_audio_bytes(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return

        encoded_audio = base64.b64encode(pcm_bytes).decode("utf-8")

        await self._send(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": encoded_audio,
                    }
                }
            }
        )

    # -------------------------------------------------------------------------
    # Text input helper
    # -------------------------------------------------------------------------

    async def _send_text_input(
        self,
        text: str,
        *,
        role: str = "USER",
        interactive: bool = False,
    ) -> None:
        content_name = str(uuid.uuid4())

        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": interactive,
                        "role": role,
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        },
                    }
                }
            }
        )

        await self._send(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": text,
                    }
                }
            }
        )

        await self._send(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def start_session(self) -> None:
        if self.is_active:
            print("[nova] Session is already active")
            return

        print("[nova] Starting session")

        # Important: _build_client is async and must be awaited.
        if self.client is None:
            await self._build_client()

        if self.client is None:
            raise RuntimeError("Failed to initialize Bedrock Runtime client")

        self.stream = (
            await self.client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=self.model_id
                )
            )
        )

        self.is_active = True

        try:
            await self._send(
                {
                    "event": {
                        "sessionStart": {
                            "inferenceConfiguration": {
                                "maxTokens": 1024,
                                "topP": 0.9,
                                "temperature": 0.7,
                            },
                            "turnDetectionConfiguration": {
                                "endpointingSensitivity": "HIGH"
                            },
                        }
                    }
                }
            )

            tool_schemas = self.mcp.list_tool_schemas()

            prompt_start_payload: dict[str, Any] = {
                "promptName": self.prompt_name,
                "textOutputConfiguration": {
                    "mediaType": "text/plain"
                },
                "audioOutputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": NOVA_OUTPUT_RATE,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "voiceId": "matthew",
                    "encoding": "base64",
                    "audioType": "SPEECH",
                },
            }

            if tool_schemas:
                prompt_start_payload["toolUseOutputConfiguration"] = {
                    "mediaType": "application/json"
                }

                prompt_start_payload["toolConfiguration"] = {
                    "tools": tool_schemas,
                    "toolChoice": {
                        "auto": {}
                    },
                }

            await self._send(
                {
                    "event": {
                        "promptStart": prompt_start_payload
                    }
                }
            )

            print(
                f"[nova] Prompt started with "
                f"{len(tool_schemas or [])} MCP tools"
            )

            # Send the system prompt.
            await self._send_text_input(
                SYSTEM_PROMPT,
                role="SYSTEM",
                interactive=False,
            )

            # Open persistent caller audio input.
            await self._open_audio_channel()

            # Start response processing before triggering the greeting.
            self.response_task = asyncio.create_task(
                self._process_responses()
            )

            # Trigger the exact greeting defined in SYSTEM_PROMPT.
            await self._send_text_input(
                "[conversation started]",
                role="USER",
                interactive=True,
            )

            print("[nova] Session setup complete")

        except Exception:
            self.is_active = False

            if self.response_task:
                self.response_task.cancel()
                self.response_task = None

            raise

    async def start_audio_input(self, call_sid: str = "") -> None:
        self._call_sid = call_sid or ""

        print(
            "[nova] Twilio audio stream connected "
            f"(CallSid={self._call_sid or 'not provided'})"
        )

        self._ready_for_audio.set()

        if call_sid:
            await self._send_text_input(
                f"[system] The active Twilio CallSid is: {call_sid}",
                role="USER",
                interactive=False,
            )

            print(f"[nova] Injected Twilio CallSid: {call_sid}")

    async def send_audio_chunk(self, pcm_bytes: bytes) -> None:
        if not self.is_active or not pcm_bytes:
            return

        self._audio_buffer.extend(pcm_bytes)

        # PCM16 mono at 16 kHz:
        # 16,000 samples/sec * 2 bytes = 32,000 bytes/sec.
        self._audio_buffer_ms += (len(pcm_bytes) * 1000) // (
            NOVA_INPUT_RATE * 2
        )

        if self._audio_buffer_ms < self.CHUNK_MS:
            return

        chunk = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._audio_buffer_ms = 0

        if self._ready_for_audio.is_set():
            await self._send_audio_bytes(chunk)

    async def end_audio_input(self) -> None:
        if self._audio_buffer and self._ready_for_audio.is_set():
            remaining_audio = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            self._audio_buffer_ms = 0

            with suppress(Exception):
                await self._send_audio_bytes(remaining_audio)

        await self._close_audio_channel()

    async def end_session(self) -> None:
        if not self.is_active:
            return

        print("[nova] Ending session")

        self.is_active = False
        self._ready_for_audio.clear()
        self.barge_in_event.clear()

        try:
            if self.stream is not None:
                with suppress(Exception):
                    await self._close_audio_channel()

                with suppress(Exception):
                    await self._send(
                        {
                            "event": {
                                "promptEnd": {
                                    "promptName": self.prompt_name
                                }
                            }
                        }
                    )

                with suppress(Exception):
                    await self._send(
                        {
                            "event": {
                                "sessionEnd": {}
                            }
                        }
                    )

                with suppress(Exception):
                    await self.stream.input_stream.close()

        finally:
            if self.response_task:
                self.response_task.cancel()

                with suppress(asyncio.CancelledError):
                    await self.response_task

                self.response_task = None

            if self.client is not None:
                with suppress(Exception):
                    await self.client.close()

                self.client = None

            self.stream = None
            self._clear_audio_queue_nowait()
            self._audio_buffer.clear()
            self._audio_buffer_ms = 0
            self._nova_speaking = False
            self._barge_in_fired = False
            self._reset_tool_state()

            print("[nova] Session ended")

    # -------------------------------------------------------------------------
    # MCP tool handling
    # -------------------------------------------------------------------------

    async def _send_tool_result(
        self,
        tool_use_id: str,
        result_content: str,
    ) -> None:
        if not tool_use_id:
            raise ValueError("tool_use_id is required")

        if not isinstance(result_content, str):
            result_content = json.dumps(
                result_content,
                ensure_ascii=False,
                default=str,
            )

        content_name = str(uuid.uuid4())

        await self._send(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "USER",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            },
                        },
                    }
                }
            }
        )

        await self._send(
            {
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": result_content,
                    }
                }
            }
        )

        await self._send(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )

    async def _execute_tool(self) -> None:
        tool_use_id = self._tool_use_id
        tool_name = self._tool_name
        tool_input = dict(self._tool_input)

        self._reset_tool_state()

        if not tool_use_id:
            print("[tool] Missing toolUseId")
            return

        if not tool_name:
            error_result = json.dumps(
                {
                    "error": "Nova requested a tool without a tool name"
                }
            )
            await self._send_tool_result(
                tool_use_id,
                error_result,
            )
            return

        print(f"[tool] Calling {tool_name} with {tool_input}")

        try:
            result = await self.mcp.call_tool(
                tool_name,
                tool_input,
            )

            if isinstance(result, str):
                result_string = result
            else:
                result_string = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

        except Exception as exc:
            print(f"[tool] {tool_name} failed: {exc}")

            result_string = json.dumps(
                {
                    "error": str(exc),
                    "tool": tool_name,
                },
                ensure_ascii=False,
            )

        await self._send_tool_result(
            tool_use_id,
            result_string,
        )

        print(f"[tool] Result sent for {tool_name}")

    # -------------------------------------------------------------------------
    # Bedrock response processor
    # -------------------------------------------------------------------------

    async def _process_responses(self) -> None:
        current_content_type: Optional[str] = None
        current_stage = ""

        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()

                if not result.value or not result.value.bytes_:
                    continue

                try:
                    data = json.loads(
                        result.value.bytes_.decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    print(f"[nova] Invalid response payload: {exc}")
                    continue

                event = data.get("event", {})

                if not event:
                    continue

                # Top-level interruption event.
                if "interrupted" in event:
                    await self._handle_barge_in(
                        "interrupted-event"
                    )
                    continue

                if "contentStart" in event:
                    content_start = event["contentStart"]

                    self._role = content_start.get("role")
                    current_content_type = content_start.get("type")
                    self._current_content_type = current_content_type

                    additional_fields = content_start.get(
                        "additionalModelFields"
                    )

                    current_stage = ""

                    if additional_fields:
                        try:
                            if isinstance(additional_fields, str):
                                parsed_fields = json.loads(
                                    additional_fields
                                )
                            elif isinstance(additional_fields, dict):
                                parsed_fields = additional_fields
                            else:
                                parsed_fields = {}

                            current_stage = parsed_fields.get(
                                "generationStage",
                                "",
                            )
                        except json.JSONDecodeError:
                            current_stage = ""

                    # FINAL text indicates completion of the assistant turn.
                    if (
                        self._role == "ASSISTANT"
                        and current_content_type == "TEXT"
                        and current_stage == "FINAL"
                    ):
                        self._nova_speaking = False
                        self._barge_in_fired = False
                        self._display_assistant_text = False
                        continue

                    # A USER content block while Nova is speaking indicates
                    # caller interruption.
                    if (
                        self._role == "USER"
                        and self._nova_speaking
                    ):
                        await self._handle_barge_in(
                            "user-started-speaking"
                        )

                    elif self._role == "ASSISTANT":
                        self._barge_in_fired = False

                    if current_content_type == "TOOL":
                        self._in_tool_use = True
                        self._tool_input = {}

                        tool_use = (
                            content_start.get("toolUseContent")
                            or content_start.get("toolUse")
                            or content_start
                        )

                        self._tool_use_id = (
                            tool_use.get("toolUseId")
                            or tool_use.get("toolInvocationId")
                            or content_start.get("toolUseId")
                            or content_start.get("toolInvocationId")
                        )

                        self._tool_name = (
                            tool_use.get("toolName")
                            or tool_use.get("name")
                            or content_start.get("toolName")
                            or content_start.get("name")
                        )

                        print(
                            "[tool] Bedrock requested "
                            f"{self._tool_name} "
                            f"(id={self._tool_use_id})"
                        )

                    else:
                        self._in_tool_use = False

                        if current_content_type == "TEXT":
                            self._display_assistant_text = (
                                current_stage == "SPECULATIVE"
                            )

                elif "toolUse" in event:
                    tool_use = event["toolUse"]

                    self._in_tool_use = True

                    if not self._tool_use_id:
                        self._tool_use_id = (
                            tool_use.get("toolUseId")
                            or tool_use.get("toolInvocationId")
                        )

                    if not self._tool_name:
                        self._tool_name = (
                            tool_use.get("toolName")
                            or tool_use.get("name")
                        )

                    raw_input = tool_use.get("content", "{}")

                    if isinstance(raw_input, dict):
                        self._tool_input = raw_input
                    elif isinstance(raw_input, str):
                        try:
                            self._tool_input = json.loads(raw_input)
                        except json.JSONDecodeError:
                            self._tool_input = {
                                "raw": raw_input
                            }
                    else:
                        self._tool_input = {
                            "raw": raw_input
                        }

                elif "contentEnd" in event:
                    if self._in_tool_use:
                        await self._execute_tool()

                    elif (
                        self._role == "ASSISTANT"
                        and current_content_type == "AUDIO"
                    ):
                        self._nova_speaking = False

                    current_content_type = None
                    current_stage = ""
                    self._current_content_type = None
                    self._display_assistant_text = False

                elif "textOutput" in event:
                    text_output = event["textOutput"]
                    text = text_output.get("content", "")

                    # Handle inline interruption content reliably.
                    interrupted = False

                    if isinstance(text, str):
                        stripped_text = text.strip()

                        try:
                            parsed_text = json.loads(stripped_text)

                            interrupted = (
                                isinstance(parsed_text, dict)
                                and parsed_text.get("interrupted") is True
                            )
                        except json.JSONDecodeError:
                            interrupted = (
                                '"interrupted"' in stripped_text
                                and "true" in stripped_text.lower()
                            )

                    if interrupted:
                        await self._handle_barge_in(
                            "interrupted-text-signal"
                        )
                        continue

                    if (
                        self._role == "ASSISTANT"
                        and self._display_assistant_text
                    ):
                        print(f"[Assistant] {text}")

                    elif self._role == "USER":
                        print(f"[User] {text}")

                elif "audioOutput" in event:
                    if self._barge_in_fired:
                        continue

                    encoded_audio = event["audioOutput"].get(
                        "content"
                    )

                    if not encoded_audio:
                        continue

                    try:
                        audio_bytes = base64.b64decode(
                            encoded_audio
                        )
                    except Exception as exc:
                        print(
                            f"[nova] Invalid audio output: {exc}"
                        )
                        continue

                    if not self._nova_speaking:
                        self._nova_speaking = True

                    await self.audio_queue.put(
                        (
                            self._generation_id,
                            audio_bytes,
                        )
                    )

        except asyncio.CancelledError:
            pass

        except Exception as exc:
            print(
                "[nova response error] "
                f"{type(exc).__name__}: {exc}"
            )

            import traceback

            traceback.print_exc()

            raw_error = (
                getattr(exc, "body", None)
                or getattr(exc, "message", None)
            )

            if raw_error:
                print(
                    "[nova response error] "
                    f"raw payload: {raw_error}"
                )

            self.is_active = False