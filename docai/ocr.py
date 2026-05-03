from __future__ import annotations

import json
import mimetypes
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OCRProviderError(RuntimeError):
    """Raised when an OCR provider cannot process a document."""


class OCRProvider(Protocol):
    def extract_pages(
        self, document_path: str | Path, pages: Sequence[int] | None = None
    ) -> dict[int, str]:
        """Return OCR text keyed by 1-based page number."""


Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], Any]


@dataclass(frozen=True)
class OCRDocument:
    page_texts: dict[int, str]
    job_id: str | None = None
    studio_link: str | None = None


class ReductoParseOCRProvider:
    """OCR provider backed by Reducto's Parse API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://platform.reducto.ai",
        timeout: float = 120.0,
        transport: Transport | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or _urllib_transport
        self._document_cache: dict[Path, OCRDocument] = {}

    def extract_pages(
        self, document_path: str | Path, pages: Sequence[int] | None = None
    ) -> dict[int, str]:
        document = self.parse_document(document_path)
        if pages is None:
            return dict(document.page_texts)

        requested_pages = {int(page) for page in pages}
        return {
            page: text
            for page, text in document.page_texts.items()
            if page in requested_pages
        }

    def parse_document(self, document_path: str | Path) -> OCRDocument:
        path = Path(document_path)
        if not path.exists():
            raise OCRProviderError(f"OCR source file does not exist: {path}")

        cache_key = path.resolve()
        cached = self._document_cache.get(cache_key)
        if cached is not None:
            return cached

        file_id = self._upload(path)
        response = self._parse(file_id)
        document = OCRDocument(
            page_texts=_page_texts_from_parse_response(response, self._fetch_json_url),
            job_id=_optional_str(response.get("job_id")),
            studio_link=_optional_str(response.get("studio_link")),
        )
        self._document_cache[cache_key] = document
        return document

    def _upload(self, path: Path) -> str:
        boundary = f"----docai-{uuid.uuid4().hex}"
        body = _multipart_file_body(path, boundary)
        headers = self._headers(
            {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            }
        )
        response = self.transport(
            "POST", f"{self.base_url}/upload", headers, body, self.timeout
        )
        file_id = response.get("file_id") if isinstance(response, Mapping) else None
        if not isinstance(file_id, str) or not file_id:
            raise OCRProviderError("Reducto upload did not return a file_id.")
        return file_id

    def _parse(self, file_id: str) -> Mapping[str, Any]:
        payload = {
            "input": file_id,
            "settings": {
                "return_ocr_data": True,
                "force_url_result": False,
            },
        }
        headers = self._headers({"Content-Type": "application/json"})
        response = self.transport(
            "POST",
            f"{self.base_url}/parse",
            headers,
            json.dumps(payload).encode("utf-8"),
            self.timeout,
        )
        if not isinstance(response, Mapping):
            raise OCRProviderError("Reducto Parse returned a non-object response.")
        return response

    def _fetch_json_url(self, url: str) -> Any:
        return self.transport("GET", url, {}, None, self.timeout)

    def _headers(self, extra: Mapping[str, str]) -> dict[str, str]:
        api_key = self.api_key or os.environ.get("REDUCTO_API_KEY")
        if not api_key:
            raise OCRProviderError(
                "REDUCTO_API_KEY is required when using the Reducto OCR provider."
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        headers.update(extra)
        return headers


def create_ocr_provider(
    provider: str | OCRProvider | None, **provider_kwargs: Any
) -> OCRProvider | None:
    if provider is None:
        return None
    if not isinstance(provider, str):
        return provider

    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized in {"reducto", "reducto_parse"}:
        return ReductoParseOCRProvider(**provider_kwargs)

    raise ValueError(f"Unsupported OCR provider: {provider}")


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> Any:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OCRProviderError(
            f"Reducto API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise OCRProviderError(f"Reducto API request failed: {exc.reason}") from exc

    try:
        return json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise OCRProviderError("Reducto API returned invalid JSON.") from exc


def _multipart_file_body(path: Path, boundary: str) -> bytes:
    filename = path.name.replace('"', '\\"')
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer


def _page_texts_from_parse_response(
    response: Mapping[str, Any], fetch_json_url: Callable[[str], Any]
) -> dict[int, str]:
    result = response.get("result", response)
    if isinstance(result, Mapping) and result.get("type") == "url":
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise OCRProviderError("Reducto URL result did not include a result URL.")
        result = fetch_json_url(url)

    if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping):
        result = result["result"]
    if isinstance(result, list):
        result = {"chunks": result}
    if not isinstance(result, Mapping):
        return {}

    pages = _ocr_line_page_texts(result)
    if pages:
        return pages
    return _chunk_page_texts(result)


def _ocr_line_page_texts(result: Mapping[str, Any]) -> dict[int, str]:
    ocr = result.get("ocr")
    if not isinstance(ocr, Mapping):
        return {}

    pages: defaultdict[int, list[str]] = defaultdict(list)
    for line in _list_items(ocr.get("lines")):
        text = _optional_str(line.get("text")).strip()
        page = _page_number(line)
        if text and page is not None:
            pages[page].append(text)

    return _compact_pages(pages)


def _chunk_page_texts(result: Mapping[str, Any]) -> dict[int, str]:
    pages: defaultdict[int, list[str]] = defaultdict(list)
    for chunk in _list_items(result.get("chunks")):
        blocks = _list_items(chunk.get("blocks"))
        if blocks:
            for block in blocks:
                text = _optional_str(block.get("content")).strip()
                page = _page_number(block)
                if text and page is not None:
                    pages[page].append(text)
            continue

        text = _optional_str(chunk.get("content")).strip()
        if text:
            pages[_page_number(chunk) or 1].append(text)

    return _compact_pages(pages)


def _list_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _page_number(item: Mapping[str, Any]) -> int | None:
    bbox = item.get("bbox")
    if isinstance(bbox, Mapping):
        page = bbox.get("page")
        if isinstance(page, int):
            return page

    for key in ("page", "page_num"):
        page = item.get(key)
        if isinstance(page, int):
            return page
    return None


def _compact_pages(pages: Mapping[int, list[str]]) -> dict[int, str]:
    return {
        page: "\n".join(part for part in parts if part).strip()
        for page, parts in sorted(pages.items())
        if any(parts)
    }


def _optional_str(value: Any) -> str:
    return value if isinstance(value, str) else ""
