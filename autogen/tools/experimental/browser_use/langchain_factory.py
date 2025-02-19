# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable

from ....import_utils import optional_import_block, skip_on_missing_imports

with optional_import_block():
    from langchain_anthropic import ChatAnthropic  # noqa
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa
    from langchain_ollama import ChatOllama
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.language_models import BaseChatModel


__all__ = ["LangchainFactory"]


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
class LangchainFactory(ABC):
    _factories: list["LangchainFactory"] = []

    @classmethod
    def create_base_chat_model(cls, llm_config: dict[str, Any]) -> BaseChatModel:  # type: ignore [no-any-unimported]
        for factory in LangchainFactory._factories:
            if factory.accepts(llm_config):
                return factory.create(llm_config)

        raise ValueError("Could not find a factory for the given config.")

    @classmethod
    def register_factory(cls) -> Callable[[type["LangchainFactory"]], type["LangchainFactory"]]:
        def decorator(factory: type["LangchainFactory"]) -> type["LangchainFactory"]:
            cls._factories.append(factory())
            return factory

        return decorator

    @classmethod
    def get_first_llm_config(cls, llm_config: dict[str, Any]) -> dict[str, Any]:
        if "config_list" not in llm_config:
            if "model" in llm_config:
                return llm_config
            raise ValueError("llm_config must be a valid config dictionary.")

        if len(llm_config["config_list"]) == 0:
            raise ValueError("Config list must contain at least one config.")

        return llm_config["config_list"][0]  # type: ignore [no-any-return]

    @classmethod
    @abstractmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:  # type: ignore [no-any-unimported]
        ...

    @classmethod
    @abstractmethod
    def get_api_type(cls) -> str: ...

    @classmethod
    def accepts(cls, llm_config: dict[str, Any]) -> bool:
        config = llm_config["config_list"][0]
        return config.get("api_type", "openai") == cls.get_api_type()  # type: ignore [no-any-return]


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
@LangchainFactory.register_factory()
class ChatOpenAIFactory(LangchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:  # type: ignore [no-any-unimported]
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")

        return ChatOpenAI(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
@LangchainFactory.register_factory()
class AzureChatOpenAIFactory(LangchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:  # type: ignore [no-any-unimported]
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")

        return AzureChatOpenAI(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"


@skip_on_missing_imports(
    ["langchain_anthropic", "langchain_google_genai", "langchain_ollama", "langchain_openai", "langchain_core"],
    "browser-use",
)
@LangchainFactory.register_factory()
class ChatOllamaFactory(LangchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:  # type: ignore [no-any-unimported]
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")
        config["base_url"] = config.pop("client_host", None)

        return ChatOllama(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"
