import os
import re
from typing import Iterator
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import logging
from rag.utils.file_reader import FileReaderFactory
from langchain_text_splitters import TokenTextSplitter


class DocumentLoader:
    def __init__(
        self,
        path: str,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
        metadata: dict = {},
        spliter=None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata = metadata
        self.spliter = spliter or TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    @staticmethod
    def if_valid_file(full_path: str) -> bool:
        if os.path.isfile(full_path) and FileReaderFactory.is_supported_file(full_path):
            return True

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        if os.path.isdir(self.path):
            for filename in os.listdir(self.path):
                full_path = os.path.join(self.path, filename)
                if self.if_valid_file(full_path):
                    yield from self.load_file(full_path)
                else:
                    logging.warning(f"Skipping {full_path}")
        else:
            yield from self.load_file(self.path)

    def add_collection(self, file_path: str):
        if "collection" not in self.metadata:
            self.metadata["collection"] = re.sub(r"[^A-Za-z0-9]", "", file_path)
        pass

    def load_file(self, file_path: str) -> Iterator[Document]:
        reader = FileReaderFactory.get_reader(file_path)
        content = reader.read(file_path)
        self.add_collection(file_path)
        if self.chunk_size:
            chunks = self.spliter.split_text(content)
            for idx, chunk in enumerate(chunks):
                self.metadata.update(
                    {"source": file_path, "length": len(chunk), "chunk": idx}
                )
                yield Document(page_content=chunk, metadata=self.metadata)
        else:
            self.metadata.update(
                {"source": file_path, "length": len(content), "chunk": 0}
            )
            yield Document(page_content=content, metadata=self.metadata)
