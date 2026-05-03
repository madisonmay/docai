from pathlib import Path

from byaldi import RAGMultiModalModel
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI

from docai.ocr import OCRProvider, OCRProviderError, create_ocr_provider


class Extractor:
    def __init__(
        self,
        index_name: str,
        ocr_provider: str | OCRProvider | None = None,
    ):
        self.rag = RAGMultiModalModel.from_index(index_name)
        self.ocr_provider = create_ocr_provider(ocr_provider)

    def extract(self, query: str, data_model: BaseModel, k: int = 3):
        results = self.rag.search(query=query, k=k)
        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{result.base64}",
                },
            }
            for result in results
        ]
        ocr_content = self._ocr_content(results)
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
        ).with_structured_output(data_model)
        response = model.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Extract structured information from this insurance policy application",
                        },
                    ]
                    + ocr_content
                    + image_content
                )
            ]
        )
        return response

    def _ocr_content(self, results):
        if self.ocr_provider is None:
            return []

        sources_by_path = self._source_pages(results)
        content = []
        for source_path, pages in sources_by_path.items():
            page_texts = self.ocr_provider.extract_pages(source_path, sorted(pages))
            for page in sorted(pages):
                text = page_texts.get(page)
                if text:
                    content.append(
                        {
                            "type": "text",
                            "text": f"OCR text from {source_path} page {page}:\n{text}",
                        }
                    )
        return content

    def _source_pages(self, results):
        doc_ids_to_file_names = self.rag.get_doc_ids_to_file_names()
        source_pages = {}
        for result in results:
            source_path = self._source_path(result.doc_id, doc_ids_to_file_names)
            if source_path is None:
                raise OCRProviderError(
                    f"Cannot run OCR because doc_id {result.doc_id} has no source file mapping."
                )
            if not source_path.exists():
                raise OCRProviderError(
                    f"Cannot run OCR because source file does not exist: {source_path}"
                )

            page_num = int(result.page_num)
            source_pages.setdefault(source_path, set()).add(page_num)
        return source_pages

    @staticmethod
    def _source_path(doc_id, doc_ids_to_file_names):
        for key in (doc_id, str(doc_id)):
            file_name = doc_ids_to_file_names.get(key)
            if file_name is not None:
                return Path(file_name)
        try:
            file_name = doc_ids_to_file_names.get(int(doc_id))
        except (TypeError, ValueError):
            file_name = None
        return Path(file_name) if file_name is not None else None
