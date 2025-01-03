# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union
from unittest.mock import MagicMock, _Call, call
from uuid import UUID

import pytest
import termcolor.termcolor

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.coding.base import CodeBlock
from autogen.messages.agent_messages import (
    ClearAgentsHistory,
    ClearConversableAgentHistory,
    ContentMessage,
    ConversableAgentUsageSummary,
    ExecuteCodeBlock,
    ExecuteFunction,
    FunctionCall,
    FunctionCallMessage,
    FunctionResponseMessage,
    GenerateCodeExecutionReply,
    GroupChatResume,
    GroupChatRunChat,
    MessageRole,
    PostCarryoverProcessing,
    SelectSpeaker,
    SpeakerAttempt,
    TerminationAndHumanReply,
    TextMessage,
    ToolCall,
    ToolCallMessage,
    ToolResponse,
    ToolResponseMessage,
    create_received_message_model,
)
from autogen.oai.client import OpenAIWrapper


@pytest.fixture(autouse=True)
def enable_color_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_can_do_colour(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(termcolor.termcolor, "_can_do_colour", mock_can_do_colour)


@pytest.fixture
def sender() -> ConversableAgent:
    return ConversableAgent("sender", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")


@pytest.fixture
def recipient() -> ConversableAgent:
    return ConversableAgent("recipient", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")


def test_tool_responses(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    message = {
        "role": "tool",
        "tool_responses": [
            {"tool_call_id": "call_rJfVpHU3MXuPRR2OAdssVqUV", "role": "tool", "content": "Timer is done!"},
            {"tool_call_id": "call_zFZVYovdsklFYgqxttcOHwlr", "role": "tool", "content": "Stopwatch is done!"},
        ],
        "content": "Timer is done!\\n\\nStopwatch is done!",
    }
    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, ToolResponseMessage)
    assert actual.role == "tool"
    assert actual.sender_name == "sender"
    assert actual.recipient_name == "recipient"
    assert actual.content == "Timer is done!\\n\\nStopwatch is done!"
    assert len(actual.tool_responses) == 2

    assert isinstance(actual.tool_responses[0], ToolResponse)
    assert actual.tool_responses[0].tool_call_id == "call_rJfVpHU3MXuPRR2OAdssVqUV"
    assert actual.tool_responses[0].role == "tool"
    assert actual.tool_responses[0].content == "Timer is done!"

    assert isinstance(actual.tool_responses[1], ToolResponse)
    assert actual.tool_responses[1].tool_call_id == "call_zFZVYovdsklFYgqxttcOHwlr"
    assert actual.tool_responses[1].role == "tool"
    assert actual.tool_responses[1].content == "Stopwatch is done!"

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("\x1b[32m***** Response from calling tool (call_rJfVpHU3MXuPRR2OAdssVqUV) *****\x1b[0m", flush=True),
        call("Timer is done!", flush=True),
        call("\x1b[32m**********************************************************************\x1b[0m", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
        call("\x1b[32m***** Response from calling tool (call_zFZVYovdsklFYgqxttcOHwlr) *****\x1b[0m", flush=True),
        call("Stopwatch is done!", flush=True),
        call("\x1b[32m**********************************************************************\x1b[0m", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


@pytest.mark.parametrize(
    "message",
    [
        {"name": "get_random_number", "role": "function", "content": "76"},
        {"name": "get_random_number", "role": "function", "content": 2},
    ],
)
def test_function_response(
    uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent, message: dict[str, Any]
) -> None:
    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, FunctionResponseMessage)

    assert actual.name == "get_random_number"
    assert actual.role == "function"
    assert actual.content == message["content"]
    assert actual.sender_name == "sender"
    assert actual.recipient_name == "recipient"

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("\x1b[32m***** Response from calling function (get_random_number) *****\x1b[0m", flush=True),
        call(message["content"], flush=True),
        call("\x1b[32m**************************************************************\x1b[0m", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


def test_function_call(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    message = {"content": "Let's play a game.", "function_call": {"name": "get_random_number", "arguments": "{}"}}

    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, FunctionCallMessage)

    assert actual.content == "Let's play a game."
    assert actual.sender_name == "sender"
    assert actual.recipient_name == "recipient"

    assert isinstance(actual.function_call, FunctionCall)
    assert actual.function_call.name == "get_random_number"
    assert actual.function_call.arguments == "{}"

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("Let's play a game.", flush=True),
        call("\x1b[32m***** Suggested function call: get_random_number *****\x1b[0m", flush=True),
        call("Arguments: \n", "{}", flush=True, sep=""),
        call("\x1b[32m******************************************************\x1b[0m", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


@pytest.mark.parametrize(
    "role",
    ["assistant", None],
)
def test_tool_calls(
    uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent, role: Optional[MessageRole]
) -> None:
    message = {
        "content": None,
        "refusal": None,
        "role": role,
        "audio": None,
        "function_call": None,
        "tool_calls": [
            {
                "id": "call_rJfVpHU3MXuPRR2OAdssVqUV",
                "function": {"arguments": '{"num_seconds": "1"}', "name": "timer"},
                "type": "function",
            },
            {
                "id": "call_zFZVYovdsklFYgqxttcOHwlr",
                "function": {"arguments": '{"num_seconds": "2"}', "name": "stopwatch"},
                "type": "function",
            },
        ],
    }

    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, ToolCallMessage)

    assert actual.content is None
    assert actual.refusal is None
    assert actual.role == role
    assert actual.audio is None
    assert actual.function_call is None
    assert actual.sender_name == "sender"
    assert actual.recipient_name == "recipient"

    assert len(actual.tool_calls) == 2

    assert isinstance(actual.tool_calls[0], ToolCall)
    assert actual.tool_calls[0].id == "call_rJfVpHU3MXuPRR2OAdssVqUV"
    assert actual.tool_calls[0].function.name == "timer"  # type: ignore [union-attr]
    assert actual.tool_calls[0].function.arguments == '{"num_seconds": "1"}'  # type: ignore [union-attr]
    assert actual.tool_calls[0].type == "function"

    assert isinstance(actual.tool_calls[1], ToolCall)
    assert actual.tool_calls[1].id == "call_zFZVYovdsklFYgqxttcOHwlr"
    assert actual.tool_calls[1].function.name == "stopwatch"  # type: ignore [union-attr]
    assert actual.tool_calls[1].function.arguments == '{"num_seconds": "2"}'  # type: ignore [union-attr]
    assert actual.tool_calls[1].type == "function"

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("\x1b[32m***** Suggested tool call (call_rJfVpHU3MXuPRR2OAdssVqUV): timer *****\x1b[0m", flush=True),
        call("Arguments: \n", '{"num_seconds": "1"}', flush=True, sep=""),
        call("\x1b[32m**********************************************************************\x1b[0m", flush=True),
        call("\x1b[32m***** Suggested tool call (call_zFZVYovdsklFYgqxttcOHwlr): stopwatch *****\x1b[0m", flush=True),
        call("Arguments: \n", '{"num_seconds": "2"}', flush=True, sep=""),
        call("\x1b[32m**************************************************************************\x1b[0m", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


def test_context_message(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    message = {"content": "hello {name}", "context": {"name": "there"}}

    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, ContentMessage)
    expected_model_dump = {
        "uuid": uuid,
        "content": "hello {name}",
        "sender_name": "sender",
        "recipient_name": "recipient",
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("hello {name}", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


def test_context_lambda_message(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    message = {
        "content": lambda context: f"hello {context['name']}",
        "context": {
            "name": "there",
        },
    }

    actual = create_received_message_model(uuid=uuid, message=message, sender=sender, recipient=recipient)

    assert isinstance(actual, ContentMessage)
    expected_model_dump = {
        "uuid": uuid,
        "content": "hello there",
        "sender_name": "sender",
        "recipient_name": "recipient",
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[33msender\x1b[0m (to recipient):\n", flush=True),
        call("hello there", flush=True),
        call(
            "\n", "--------------------------------------------------------------------------------", flush=True, sep=""
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


def test_PostCarryoverProcessing(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    chat_info = {
        "carryover": ["This is a test message 1", "This is a test message 2"],
        "message": "Start chat",
        "verbose": True,
        "sender": sender,
        "recipient": recipient,
        "summary_method": "last_msg",
        "max_turns": 5,
    }

    actual = PostCarryoverProcessing(uuid=uuid, chat_info=chat_info)
    assert isinstance(actual, PostCarryoverProcessing)

    expected = {
        "uuid": uuid,
        "carryover": ["This is a test message 1", "This is a test message 2"],
        "message": "Start chat",
        "verbose": True,
        "sender_name": "sender",
        "recipient_name": "recipient",
        "summary_method": "last_msg",
        "summary_args": None,
        "max_turns": 5,
    }
    assert actual.model_dump() == expected, f"{actual.model_dump()} != {expected}"

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call(
            "\x1b[34m\n********************************************************************************\x1b[0m",
            flush=True,
            sep="",
        ),
        call("\x1b[34mStarting a new chat....\x1b[0m", flush=True),
        call("\x1b[34mMessage:\nStart chat\x1b[0m", flush=True),
        call("\x1b[34mCarryover:\nThis is a test message 1\nThis is a test message 2\x1b[0m", flush=True),
        call(
            "\x1b[34m\n********************************************************************************\x1b[0m",
            flush=True,
            sep="",
        ),
    ]

    assert mock.call_args_list == expected_call_args_list


@pytest.mark.parametrize(
    "carryover, expected",
    [
        ("This is a test message 1", "This is a test message 1"),
        (
            ["This is a test message 1", "This is a test message 2"],
            "This is a test message 1\nThis is a test message 2",
        ),
        (
            [
                {"content": "This is a test message 1"},
                {"content": "This is a test message 2"},
            ],
            "This is a test message 1\nThis is a test message 2",
        ),
        ([1, 2, 3], "1\n2\n3"),
    ],
)
def test__process_carryover(
    carryover: Union[str, list[Union[str, dict[str, Any], Any]]],
    expected: str,
    uuid: UUID,
    sender: ConversableAgent,
    recipient: ConversableAgent,
) -> None:
    chat_info = {
        "carryover": carryover,
        "message": "Start chat",
        "verbose": True,
        "sender": sender,
        "recipient": recipient,
        "summary_method": "last_msg",
        "max_turns": 5,
    }

    post_carryover_processing = PostCarryoverProcessing(uuid=uuid, chat_info=chat_info)
    expected_model_dump = {
        "uuid": uuid,
        "carryover": carryover,
        "message": "Start chat",
        "verbose": True,
        "sender_name": "sender",
        "recipient_name": "recipient",
        "summary_method": "last_msg",
        "summary_args": None,
        "max_turns": 5,
    }
    assert post_carryover_processing.model_dump() == expected_model_dump

    actual = post_carryover_processing._process_carryover()
    assert actual == expected


@pytest.mark.parametrize(
    "agent, nr_messages_to_preserve, expected",
    [
        (None, None, "Clearing history for all agents."),
        (None, 5, "Clearing history for all agents except last 5 messages."),
        (
            ConversableAgent("clear_agent", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
            None,
            "Clearing history for clear_agent.",
        ),
        (
            ConversableAgent("clear_agent", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
            5,
            "Clearing history for clear_agent except last 5 messages.",
        ),
    ],
)
def test_ClearAgentsHistory(
    agent: Optional[ConversableAgent], nr_messages_to_preserve: Optional[int], expected: str, uuid: UUID
) -> None:
    actual = ClearAgentsHistory(uuid=uuid, agent=agent, nr_messages_to_preserve=nr_messages_to_preserve)
    assert isinstance(actual, ClearAgentsHistory)

    expected_model_dump = {
        "uuid": uuid,
        "agent_name": "clear_agent" if agent else None,
        "nr_messages_to_preserve": nr_messages_to_preserve,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [call(expected)]
    assert mock.call_args_list == expected_call_args_list


@pytest.mark.parametrize(
    "mentions, expected",
    [
        ({"agent_1": 1}, "\x1b[32m>>>>>>>> Select speaker attempt 1 of 3 successfully selected: agent_1\x1b[0m"),
        (
            {"agent_1": 1, "agent_2": 2},
            "\x1b[31m>>>>>>>> Select speaker attempt 1 of 3 failed as it included multiple agent names.\x1b[0m",
        ),
        ({}, "\x1b[31m>>>>>>>> Select speaker attempt #1 failed as it did not include any agent names.\x1b[0m"),
    ],
)
def test_SpeakerAttempt(mentions: dict[str, int], expected: str, uuid: UUID) -> None:
    attempt = 1
    attempts_left = 2
    verbose = True

    actual = SpeakerAttempt(
        uuid=uuid, mentions=mentions, attempt=attempt, attempts_left=attempts_left, select_speaker_auto_verbose=verbose
    )
    assert isinstance(actual, SpeakerAttempt)

    expected_model_dump = {
        "uuid": uuid,
        "mentions": mentions,
        "attempt": attempt,
        "attempts_left": attempts_left,
        "verbose": verbose,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [call(expected, flush=True)]

    assert mock.call_args_list == expected_call_args_list


def test_GroupChatResume(uuid: UUID) -> None:
    last_speaker_name = "Coder"
    messages = [
        {"content": "You are an expert at coding.", "role": "system", "name": "chat_manager"},
        {"content": "Let's get coding, should I use Python?", "name": "Coder", "role": "assistant"},
    ]
    silent = False

    actual = GroupChatResume(uuid=uuid, last_speaker_name=last_speaker_name, messages=messages, silent=silent)
    assert isinstance(actual, GroupChatResume)

    expected_model_dump = {
        "uuid": uuid,
        "last_speaker_name": last_speaker_name,
        "messages": messages,
        "verbose": True,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("Prepared group chat with 2 messages, the last speaker is", "\x1b[33mCoder\x1b[0m", flush=True)
    ]

    assert mock.call_args_list == expected_call_args_list


def test_GroupChatRunChat(uuid: UUID) -> None:
    speaker = ConversableAgent(
        "assistant uno", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"
    )
    silent = False

    actual = GroupChatRunChat(uuid=uuid, speaker=speaker, silent=silent)
    assert isinstance(actual, GroupChatRunChat)

    expected_model_dump = {
        "uuid": uuid,
        "speaker_name": "assistant uno",
        "verbose": True,
    }
    assert actual.model_dump() == expected_model_dump

    assert actual.speaker_name == "assistant uno"
    assert actual.verbose is True

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [call("\x1b[32m\nNext speaker: assistant uno\n\x1b[0m", flush=True)]

    assert mock.call_args_list == expected_call_args_list


def test_TerminationAndHumanReply(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    no_human_input_msg = "NO HUMAN INPUT RECEIVED."
    human_input_mode = "ALWAYS"

    actual = TerminationAndHumanReply(
        uuid=uuid,
        no_human_input_msg=no_human_input_msg,
        human_input_mode=human_input_mode,
        sender=sender,
        recipient=recipient,
    )
    assert isinstance(actual, TerminationAndHumanReply)

    expected_model_dump = {
        "uuid": uuid,
        "no_human_input_msg": no_human_input_msg,
        "human_input_mode": human_input_mode,
        "sender_name": "sender",
        "recipient_name": "recipient",
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print_no_human_input_msg(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [call("\x1b[31m\n>>>>>>>> NO HUMAN INPUT RECEIVED.\x1b[0m", flush=True)]
    assert mock.call_args_list == expected_call_args_list

    mock = MagicMock()
    actual.print_human_input_mode(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [call("\x1b[31m\n>>>>>>>> USING AUTO REPLY...\x1b[0m", flush=True)]
    assert mock.call_args_list == expected_call_args_list


def test_ExecuteCodeBlock(uuid: UUID, sender: ConversableAgent, recipient: ConversableAgent) -> None:
    code = """print("hello world")"""
    language = "python"
    code_block_count = 0

    actual = ExecuteCodeBlock(
        uuid=uuid, code=code, language=language, code_block_count=code_block_count, recipient=recipient
    )
    assert isinstance(actual, ExecuteCodeBlock)

    expected_model_dump = {
        "uuid": uuid,
        "code": code,
        "language": language,
        "code_block_count": code_block_count,
        "recipient_name": "recipient",
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    expected_call_args_list = [
        call("\x1b[31m\n>>>>>>>> EXECUTING CODE BLOCK 0 (inferred language is python)...\x1b[0m", flush=True)
    ]

    assert mock.call_args_list == expected_call_args_list


def test_ExecuteFunction(uuid: UUID, recipient: ConversableAgent) -> None:
    func_name = "add_num"
    verbose = True

    actual = ExecuteFunction(uuid=uuid, func_name=func_name, recipient=recipient, verbose=verbose)
    assert isinstance(actual, ExecuteFunction)

    expected_model_dump = {
        "uuid": uuid,
        "func_name": func_name,
        "recipient_name": "recipient",
        "verbose": verbose,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print_executing_func(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [call("\x1b[35m\n>>>>>>>> EXECUTING FUNCTION add_num...\x1b[0m", flush=True)]
    assert mock.call_args_list == expected_call_args_list

    arguments = {"num_to_be_added": 5}
    content = "15"
    mock = MagicMock()
    actual.print_arguments_and_content(arguments, content, f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [
        call("\x1b[35m\nInput arguments: {'num_to_be_added': 5}\nOutput:\n15\x1b[0m", flush=True)
    ]
    assert mock.call_args_list == expected_call_args_list


def test_SelectSpeaker(uuid: UUID) -> None:
    agents = [
        ConversableAgent("bob", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
        ConversableAgent("charlie", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER"),
    ]

    actual = SelectSpeaker(uuid=uuid, agents=agents)  # type: ignore [arg-type]
    assert isinstance(actual, SelectSpeaker)

    expected_model_dump = {
        "uuid": uuid,
        "agent_names": ["bob", "charlie"],
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print_select_speaker(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [
        call("Please select the next speaker from the following list:"),
        call("1: bob"),
        call("2: charlie"),
    ]
    assert mock.call_args_list == expected_call_args_list

    mock = MagicMock()
    actual.print_try_count_exceeded(try_count=3, f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [call("You have tried 3 times. The next speaker will be selected automatically.")]
    assert mock.call_args_list == expected_call_args_list

    mock = MagicMock()
    actual.print_invalid_input(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [call("Invalid input. Please enter a number between 1 and 2.")]
    assert mock.call_args_list == expected_call_args_list


def test_ClearConversableAgentHistory(uuid: UUID, recipient: ConversableAgent) -> None:
    nr_messages_to_preserve = 5

    actual = ClearConversableAgentHistory(uuid=uuid, agent=recipient, nr_messages_to_preserve=nr_messages_to_preserve)
    assert isinstance(actual, ClearConversableAgentHistory)

    expected_model_dump = {
        "uuid": uuid,
        "agent_name": "recipient",
        "nr_messages_to_preserve": nr_messages_to_preserve,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print_preserving_message(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [
        call("Preserving one more message for recipient to not divide history between tool call and tool response.")
    ]
    assert mock.call_args_list == expected_call_args_list

    mock = MagicMock()
    actual.print_warning(f=mock)
    # print(mock.call_args_list)
    expected_call_args_list = [
        call(
            "\x1b[33mWARNING: `nr_preserved_messages` is ignored when clearing chat history with a specific agent.\x1b[0m",
            flush=True,
        )
    ]
    assert mock.call_args_list == expected_call_args_list


@pytest.mark.parametrize(
    "code_blocks, expected",
    [
        (
            [
                CodeBlock(code="print('hello world')", language="python"),
            ],
            [call("\x1b[31m\n>>>>>>>> EXECUTING CODE BLOCK (inferred language is python)...\x1b[0m", flush=True)],
        ),
        (
            [
                CodeBlock(code="print('hello world')", language="python"),
                CodeBlock(code="print('goodbye world')", language="python"),
            ],
            [
                call(
                    "\x1b[31m\n>>>>>>>> EXECUTING 2 CODE BLOCKS (inferred languages are [python, python])...\x1b[0m",
                    flush=True,
                )
            ],
        ),
    ],
)
def test_GenerateCodeExecutionReply(
    code_blocks: list[CodeBlock],
    expected: list[_Call],
    uuid: UUID,
    sender: ConversableAgent,
    recipient: ConversableAgent,
) -> None:
    actual = GenerateCodeExecutionReply(uuid=uuid, sender=sender, recipient=recipient)
    assert isinstance(actual, GenerateCodeExecutionReply)

    expected_model_dump = {
        "uuid": uuid,
        "sender_name": "sender",
        "recipient_name": "recipient",
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print_executing_code_block(code_blocks=code_blocks, f=mock)

    # print(mock.call_args_list)

    assert mock.call_args_list == expected


@pytest.mark.parametrize(
    "client, is_client_empty, expected",
    [
        (OpenAIWrapper(api_key="dummy api key"), False, [call("Agent 'recipient':")]),
        (None, True, [call("No cost incurred from agent 'recipient'.")]),
    ],
)
def test_ConversableAgentUsageSummary(
    client: Optional[OpenAIWrapper],
    is_client_empty: bool,
    expected: list[_Call],
    uuid: UUID,
    recipient: ConversableAgent,
) -> None:
    actual = ConversableAgentUsageSummary(uuid=uuid, recipient=recipient, client=client)
    assert isinstance(actual, ConversableAgentUsageSummary)

    expected_model_dump = {
        "uuid": uuid,
        "recipient_name": "recipient",
        "is_client_empty": is_client_empty,
    }
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(f=mock)

    # print(mock.call_args_list)

    assert mock.call_args_list == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", [call("Hello, World!")]),
        ("Over and out!", [call("Over and out!")]),
    ],
)
def test_TextMessage(text: str, expected: list[_Call], uuid: UUID) -> None:
    actual = TextMessage(uuid=uuid)
    expected_model_dump = {"uuid": uuid}
    assert isinstance(actual, TextMessage)
    assert actual.model_dump() == expected_model_dump

    mock = MagicMock()
    actual.print(text, f=mock)

    # print(mock.call_args_list)

    assert mock.call_args_list == expected
