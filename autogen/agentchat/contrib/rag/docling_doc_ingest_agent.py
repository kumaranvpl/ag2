# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Literal, Optional, Union

from autogen import ConversableAgent
from autogen.agentchat.contrib.rag.parser_utils import docling_parse_docs
from autogen.agentchat.contrib.swarm_agent import SwarmResult

DOCLING_PARSE_TOOL_NAME = "docling_parse_docs"

DEFALT_DOCLING_PARSER_PROMPT = f"""
You are an expert in parsing and understanding text. You can use {DOCLING_PARSE_TOOL_NAME} tool to parse various documents and extract information from them.
"""


class DoclingDocIngestAgent(ConversableAgent):
    """
    A DoclingDocIngestAgent is a swarm agent that ingests documents using the docling_parse_docs tool.
    """

    def __init__(
        self,
        name: str = "DoclingDocIngestAgent",
        llm_config: Optional[Union[dict, Literal[False]]] = None,
        parsed_docs_path: Union[Path, str] = "./output_dir",
    ):
        def data_ingest_task(context_variables: dict) -> SwarmResult:
            tasks = self.get_context("DocumentsToIngest")
            while tasks:
                task = tasks.pop()
                docling_parse_docs(
                    input_file_path=task["path_or_url"],
                    output_dir_path=parsed_docs_path,
                )
            print("docling init:", self._context_variables, "\n", context_variables)

            return SwarmResult(
                agent="TaskManagerAgent", value="Ingested data", context_variables=self._context_variables
            )

        super().__init__(
            name=name,
            llm_config=llm_config,
            functions=[data_ingest_task],
            system_message=DEFALT_DOCLING_PARSER_PROMPT,
        )
