import csv
import PyPDF2
import logging


class FileReader:
    def read(self, file_path: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")


class TextFileReader(FileReader):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read()


class CsvFileReader(FileReader):
    def read(self, file_path: str) -> str:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            return "\n".join([",".join(row) for row in reader])


class PdfFileReader(FileReader):
    def read(self, file_path: str) -> str:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(
                page.extract_text() or ""
                for page in pdf_reader.pages
                if page.extract_text()
            )


class FileReaderFactory:
    @staticmethod
    def get_reader(file_path: str) -> FileReader:
        if file_path.endswith(".txt"):
            return TextFileReader()
        elif file_path.endswith(".csv"):
            return CsvFileReader()
        elif file_path.endswith(".pdf"):
            return PdfFileReader()
        elif file_path.endswith(".json"):
            return TextFileReader()
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        return file_path.endswith((".txt", ".csv", ".pdf", ".json"))
