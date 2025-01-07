# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Create an OpenAI-compatible client using Cohere's API.

Example:
    ```python
    llm_config={
        "config_list": [{
            "api_type": "cohere",
            "model": "command-r-plus",
            "api_key": os.environ.get("COHERE_API_KEY")
            "client_name": "autogen-cohere", # Optional parameter
            }
    ]}

    agent = autogen.AssistantAgent("my_agent", llm_config=llm_config)
    ```

Install Cohere's python library using: pip install --upgrade cohere

Resources:
- https://docs.cohere.com/reference/chat
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

from cohere import ClientV2 as CohereV2
from cohere.types import ToolResult
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.completion_usage import CompletionUsage

from autogen.oai.client_utils import logging_formatter, validate_parameter

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add the console handler.
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logging_formatter)
    logger.addHandler(_ch)


COHERE_PRICING_1K = {
    "command-r-plus": (0.003, 0.015),
    "command-r": (0.0005, 0.0015),
    "command-nightly": (0.00025, 0.00125),
    "command": (0.015, 0.075),
    "command-light": (0.008, 0.024),
    "command-light-nightly": (0.008, 0.024),
}


class CohereClient:
    """Client for Cohere's API."""

    def __init__(self, **kwargs):
        """Requires api_key or environment variable to be set

        Args:
            api_key (str): The API key for using Cohere (or environment variable COHERE_API_KEY needs to be set)
        """
        # Ensure we have the api_key upon instantiation
        self.api_key = kwargs.get("api_key", None)
        if not self.api_key:
            self.api_key = os.getenv("COHERE_API_KEY")

        assert (
            self.api_key
        ), "Please include the api_key in your config list entry for Cohere or set the COHERE_API_KEY env variable."

        if "response_format" in kwargs and kwargs["response_format"] is not None:
            warnings.warn("response_format is not supported for Cohere, it will be ignored.", UserWarning)

    def message_retrieval(self, response) -> list:
        """
        Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message for choice in response.choices]

    def cost(self, response) -> float:
        return response.cost

    @staticmethod
    def get_usage(response) -> dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        # ...  # pragma: no cover
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model,
        }

    def parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Loads the parameters for Cohere API from the passed in parameters and returns a validated set. Checks types, ranges, and sets defaults"""
        cohere_params = {}

        # Check that we have what we need to use Cohere's API
        # We won't enforce the available models as they are likely to change
        cohere_params["model"] = params.get("model", None)
        assert cohere_params[
            "model"
        ], "Please specify the 'model' in your config list entry to nominate the Cohere model to use."

        # Validate allowed Cohere parameters
        # https://docs.cohere.com/reference/chat
        if "temperature" in params:
            cohere_params["temperature"] = validate_parameter(
                params, "temperature", (int, float), False, 0.3, (0, None), None
            )

        if "max_tokens" in params:
            cohere_params["max_tokens"] = validate_parameter(params, "max_tokens", int, True, None, (0, None), None)

        if "k" in params:
            cohere_params["k"] = validate_parameter(params, "k", int, False, 0, (0, 500), None)

        if "p" in params:
            cohere_params["p"] = validate_parameter(params, "p", (int, float), False, 0.75, (0.01, 0.99), None)

        if "seed" in params:
            cohere_params["seed"] = validate_parameter(params, "seed", int, True, None, None, None)

        if "frequency_penalty" in params:
            cohere_params["frequency_penalty"] = validate_parameter(
                params, "frequency_penalty", (int, float), True, 0, (0, 1), None
            )

        if "presence_penalty" in params:
            cohere_params["presence_penalty"] = validate_parameter(
                params, "presence_penalty", (int, float), True, 0, (0, 1), None
            )

        return cohere_params

    def create(self, params: dict) -> ChatCompletion:
        messages = params.get("messages", [])
        client_name = params.get("client_name") or "AG2"

        # Parse parameters to the Cohere API's parameters
        cohere_params = self.parse_params(params)

        cohere_params["messages"] = messages

        # Strip out name
        for message in cohere_params["messages"]:
            if "name" in message:
                del message["name"]

        if "tools" in params:
            cohere_params["tools"] = params["tools"]

        # We use chat model by default
        client = CohereV2(api_key=self.api_key, client_name=client_name)

        # Token counts will be returned
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Stream if in parameters
        streaming = True if "stream" in params and params["stream"] else False
        cohere_finish = "stop"
        tool_calls = None
        ans = None
        if streaming:
            response = client.chat_stream(**cohere_params)
            # Streaming...
            ans = ""
            for event in response:
                if event.event_type == "text-generation":
                    ans = ans + event.text
                elif event.event_type == "tool-calls-generation":
                    # When streaming, tool calls are compiled at the end into a single event_type
                    ans = event.text
                    cohere_finish = "tool_calls"
                    tool_calls = []
                    for tool_call in event.tool_calls:
                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=str(random.randint(0, 100000)),
                                function={
                                    "name": tool_call.name,
                                    "arguments": (
                                        "" if tool_call.parameters is None else json.dumps(tool_call.parameters)
                                    ),
                                },
                                type="function",
                            )
                        )

            # Not using billed_units, but that may be better for cost purposes
            prompt_tokens = event.response.meta.tokens.input_tokens
            completion_tokens = event.response.meta.tokens.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            response_id = event.response.response_id
        else:
            response = client.chat(**cohere_params)

            if response.message.tool_calls is not None:
                cohere_finish = "tool_calls"
                tool_calls = []
                for tool_call in response.message.tool_calls:

                    # if parameters are null, clear them out (Cohere can return a string "null" if no parameter values)

                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call.id,
                            function={
                                "name": tool_call.function.name,
                                "arguments": (
                                    "" if tool_call.function.arguments is None else tool_call.function.arguments
                                ),
                            },
                            type="function",
                        )
                    )
            else:
                ans: str = response.message.content[0].text

            # Not using billed_units, but that may be better for cost purposes
            prompt_tokens = response.usage.tokens.input_tokens
            completion_tokens = response.usage.tokens.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            response_id = response.id

        # 3. convert output
        message = ChatCompletionMessage(
            role="assistant",
            content=ans,
            function_call=None,
            tool_calls=tool_calls,
        )
        choices = [Choice(finish_reason=cohere_finish, index=0, message=message)]

        response_oai = ChatCompletion(
            id=response_id,
            model=cohere_params["model"],
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            cost=calculate_cohere_cost(prompt_tokens, completion_tokens, cohere_params["model"]),
        )

        return response_oai


def extract_to_cohere_tool_results(tool_call_id: str, content_output: str, all_tool_calls) -> list[dict[str, Any]]:
    temp_tool_results = []

    for tool_call in all_tool_calls:
        if tool_call["id"] == tool_call_id:

            call = {
                "name": tool_call["function"]["name"],
                "parameters": json.loads(
                    tool_call["function"]["arguments"] if not tool_call["function"]["arguments"] == "" else "{}"
                ),
            }
            output = [{"value": content_output}]
            temp_tool_results.append(ToolResult(call=call, outputs=output))
    return temp_tool_results


def calculate_cohere_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of the completion using the Cohere pricing."""
    total = 0.0

    if model in COHERE_PRICING_1K:
        input_cost_per_k, output_cost_per_k = COHERE_PRICING_1K[model]
        input_cost = (input_tokens / 1000) * input_cost_per_k
        output_cost = (output_tokens / 1000) * output_cost_per_k
        total = input_cost + output_cost
    else:
        warnings.warn(f"Cost calculation not available for {model} model", UserWarning)

    return total


class CohereError(Exception):
    """Base class for other Cohere exceptions"""

    pass


class CohereRateLimitError(CohereError):
    """Raised when rate limit is exceeded"""

    pass
