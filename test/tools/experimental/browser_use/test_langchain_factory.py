# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from autogen.import_utils import skip_on_missing_imports
from autogen.tools.experimental.browser_use.langchain_factory import LangchainFactory

# with optional_import_block():
#     from langchain_core.language_models import BaseChatModel
#     from langchain_ollama import ChatOllama
#     from langchain_openai import AzureChatOpenAI, ChatOpenAI


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
class TestLangchainFactory:
    @pytest.mark.parametrize(
        ("llm_config", "expected"),
        [
            (
                {"model": "gpt-4o-mini", "api_key": "test"},
                {"model": "gpt-4o-mini", "api_key": "test"},
            ),
            (
                {"config_list": [{"model": "gpt-4o-mini", "api_key": "test"}]},
                {"model": "gpt-4o-mini", "api_key": "test"},
            ),
            (
                {"config_list": [{"model": "gpt-4o-mini", "api_key": "test"}, {"model": "gpt-4o", "api_key": "test2"}]},
                {"model": "gpt-4o-mini", "api_key": "test"},
            ),
        ],
    )
    def test_get_first_llm_config(self, llm_config: dict[str, Any], expected: dict[str, Any]) -> None:
        assert LangchainFactory.get_first_llm_config(llm_config) == expected

    @pytest.mark.parametrize(
        ("llm_config", "error_message"),
        [
            ({}, "llm_config must be a valid config dictionary."),
            ({"config_list": []}, "Config list must contain at least one config."),
        ],
    )
    def test_get_first_llm_config_incorrect_config(self, llm_config: dict[str, Any], error_message: str) -> None:
        with pytest.raises(ValueError, match=error_message):
            LangchainFactory.get_first_llm_config(llm_config)


# @skip_on_missing_imports(
#     ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
#     "browser-use",
# )
# class TestChatOpenAIFactory:
#     @pytest.mark.parametrize(
#         ("llm_config", "expected"),
#         [
#             (
#                 {"model": "gpt-4o-mini", "api_key": "test"},
#                 {"model": "gpt-4o-mini", "api_key": "test"},
#             ),
#             (
#                 {"config_list": [{"model": "gpt-4o-mini", "api_key": "test"}]},
#                 {"model": "gpt-4o-mini", "api_key": "test"},
#             ),
#         ],
#     )
#     def test_create(self, llm_config: dict[str, Any], expected: dict[str, Any]) -> None:
#         pass
# assert isinstance(LangchainFactory.create(llm_config), ChatOpenAI)
