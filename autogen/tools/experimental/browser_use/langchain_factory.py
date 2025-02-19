# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from ....doc_utils import export_module
from ....import_utils import optional_import_block, require_optional_import
from ... import Depends, Tool
from ...dependency_injection import on

with optional_import_block():
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.language_models import BaseChatModel

class LanchainFactory(ABC):
    _factories: list["LanchainFactory"] = []
    
    @classmethod
    def create_base_chat_model(cls, llm_config: dict[str, Any]) -> BaseChatModel:
        all_factories: list[LanchainFactory] = []
        for factory in all_factories:
            if factory.accepts(llm_config):
                return factory.create(llm_config)
            
        raise ValueError(f"Could not find a factory for the given config.")
    
    @classmethod
    def register_factory(cls):
        def decorator(factory: "LanchainFactory") -> "LanchainFactory":
            cls._factories.append(factory)
            return factory
        return decorator

    @abstractmethod
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel: ...

    @abstractmethod
    @classmethod
    def get_api_type(cls) -> str: ...

    @classmethod
    def accepts(cls, llm_config: dict[str, Any]) -> bool:
        config = llm_config["config_list"][0]
        return config.get("api_type", "openai") == cls.get_api_type()

@LanchainFactory.register_factory()
class ChatOpenAIFactory(LanchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")
        
        return ChatOpenAI(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"

@LanchainFactory.register_factory()
class AzureChatOpenAIFactory(LanchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")
        
        return AzureChatOpenAI(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"


@LanchainFactory.register_factory()
class ChatOllamaFactory(LanchainFactory):
    @classmethod
    def create(cls, llm_config: dict[str, Any]) -> BaseChatModel:
        config = llm_config["config_list"][0]
        config.pop("api_type", "openai")
        config["base_url"] = config.pop("client_host", None)
        
        return ChatOllama(**config)

    @classmethod
    def get_api_type(cls) -> str:
        return "ollama"

