# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Optional

from autogen.import_utils import optional_import_block, require_optional_import

with optional_import_block():
    import chromadb
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import EmbeddingFunction
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

DEFAULT_COLLECTION_NAME = "docling-parsed-docs"

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@require_optional_import(["chromadb", "llama_index"], "rag")
class DoclingMdQueryEngine:
    """
    This query engine leverages LlamaIndex's VectorStoreIndex in combination with Chromadb to query Markdown files processed by docling.
    You can specify the collection name when initializing the database via the init_db() method and later add more documents to the collection.
    Data is persisted using Chromadb's PersistentClient, which saves your collections to a directory relative to your current path.
    LlamaIndex's VectorStoreIndex is then used to efficiently index and query the documents.
    """

    def __init__(  # type: ignore
        self,
        db_path: Optional[str] = None,
        embedding_function: "Optional[EmbeddingFunction[Any]]" = None,
        metadata: Optional[dict[str, Any]] = None,
        llm: Optional["LLM"] = None,
    ) -> None:
        """
        Initializes the DoclingMdQueryEngine with db_path, metadata, and embedding function and llm.
        Args:
            db_path: the path to save chromadb data
            embedding_function: The embedding function to use. Default embedding uses Sentence Transformers model all-MiniLM-L6-v2.
                For more embeddings that ChromaDB support, please refer to [embeddings](https://docs.trychroma.com/docs/embeddings/embedding-functions)
            metadata: The metadata used by Chromadb collection creation. Chromadb uses HNSW indexing algorithm under the hood.
                For more details about the default metadata, please refer to [HNSW configuration](https://cookbook.chromadb.dev/core/configuration/#hnsw-configuration)
            llm: LLM model used by LlamaIndex. You can find more supported LLMs at [LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)
        """
        self.llm: LLM = llm or OpenAI(model="gpt-4o", temperature=0.0)  # type: ignore[no-any-unimported]
        self.embedding_function: EmbeddingFunction[Any] = embedding_function or DefaultEmbeddingFunction()  # type: ignore[no-any-unimported,assignment]
        self.metadata: dict[str, Any] = metadata or {
            "hnsw:space": "ip",
            "hnsw:construction_ef": 30,
            "hnsw:M": 32,
        }
        self.client = chromadb.PersistentClient(path=db_path or "./chroma")

    def init_db(
        self,
        input_dir: Optional[str] = None,
        input_doc_paths: Optional[list[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize VectorDB by creating collection using given name,
        loading docs and creating index
        """
        input_dir = input_dir or ""
        input_doc_paths = input_doc_paths or []
        collection_name = collection_name or DEFAULT_COLLECTION_NAME

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata=self.metadata,
            get_or_create=True,  # If collection already exists, get the collection
        )
        logger.info(f"Collection {collection_name} was created in the database.")

        documents = self._load_doc(input_dir, input_doc_paths)
        logger.info("Documents are loaded successfully.")

        self.index = self._create_index(self.collection, documents)
        logger.info("VectorDB index was created with input documents")

    def query(self, question: str) -> str:
        """
        Query your Docling parsed md files.
        """
        self.query_engine = self.index.as_query_engine(llm=self.llm)
        response = self.query_engine.query(question)

        return str(response)

    def add_docs(self, new_doc_dir: Optional[str], new_doc_paths: Optional[list[str]]) -> None:
        """
        Add new documents to the index from either a directory or a list of file paths.
        """
        new_docs = self._load_doc(input_dir=new_doc_dir, input_docs=new_doc_paths)
        for doc in new_docs:
            self.index.insert(doc)

    def _load_doc(  # type: ignore
        self, input_dir: Optional[str], input_docs: Optional[list[str]]
    ) -> list["LlamaDocument"]:
        """
        Load documents from the input directory and/or a list of input files, if provided.
        It supports lots of [formats](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/#supported-file-types),
          but you should use Docling parsed Markdown files for this query engine.
        """
        loaded_documents = []
        if input_dir:
            logger.info(f"Loading docs from directory: {input_dir}")
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory not found: {input_dir}")
            loaded_documents.extend(SimpleDirectoryReader(input_dir=input_dir).load_data())

        if input_docs:
            for doc in input_docs:
                logger.info(f"Loading input doc: {doc}")
                if not os.path.exists(doc):
                    raise ValueError(f"Document file not found: {doc}")
            loaded_documents.extend(SimpleDirectoryReader(input_files=input_docs).load_data())

        if not input_dir and not input_docs:
            raise ValueError("No input directory or docs provided!")

        return loaded_documents

    def _create_index(  # type: ignore
        self, collection: "Collection", docs: list["LlamaDocument"]
    ) -> "VectorStoreIndex":
        """
        Create a LlamaIndex VectorStoreIndex using the provided documents and Chromadb collection.
        """

        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_documents(docs, storage_context=self.storage_context)

        return index
