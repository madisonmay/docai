from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docai.ocr import OCRProviderError, ReductoParseOCRProvider, create_ocr_provider


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": body,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ReductoParseOCRProviderTest(unittest.TestCase):
    def test_extract_pages_groups_reducto_ocr_lines_by_requested_page(self):
        transport = FakeTransport(
            [
                {"file_id": "reducto://uploaded-file"},
                {
                    "job_id": "job-123",
                    "studio_link": "https://studio.reducto.ai/job/job-123",
                    "result": {
                        "type": "full",
                        "ocr": {
                            "lines": [
                                {
                                    "text": "ignored page",
                                    "bbox": {"page": 1},
                                },
                                {
                                    "text": "first line",
                                    "bbox": {"page": 2},
                                },
                                {
                                    "text": "second line",
                                    "bbox": {"page": 2},
                                },
                            ]
                        },
                    },
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            document_path = Path(temp_dir) / "sample.pdf"
            document_path.write_bytes(b"%PDF-1.4")

            provider = ReductoParseOCRProvider(
                api_key="test-key", transport=transport
            )
            self.assertEqual(
                provider.extract_pages(document_path, pages=[2]),
                {2: "first line\nsecond line"},
            )

        upload_call, parse_call = transport.calls
        self.assertEqual(upload_call["method"], "POST")
        self.assertTrue(upload_call["url"].endswith("/upload"))
        self.assertEqual(parse_call["method"], "POST")
        self.assertTrue(parse_call["url"].endswith("/parse"))
        self.assertEqual(
            parse_call["headers"]["Authorization"], "Bearer test-key"
        )
        parse_payload = json.loads(parse_call["body"].decode("utf-8"))
        self.assertEqual(parse_payload["input"], "reducto://uploaded-file")
        self.assertTrue(parse_payload["settings"]["return_ocr_data"])

    def test_extract_pages_handles_url_results_and_chunk_fallback(self):
        transport = FakeTransport(
            [
                {"file_id": "reducto://uploaded-file"},
                {
                    "result": {
                        "type": "url",
                        "url": "https://example.test/reducto-result.json",
                    },
                },
                {
                    "chunks": [
                        {
                            "blocks": [
                                {
                                    "content": "block one",
                                    "bbox": {"page": 3},
                                },
                                {
                                    "content": "block two",
                                    "bbox": {"page": 3},
                                },
                            ]
                        }
                    ]
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            document_path = Path(temp_dir) / "sample.pdf"
            document_path.write_bytes(b"%PDF-1.4")

            provider = ReductoParseOCRProvider(
                api_key="test-key", transport=transport
            )
            self.assertEqual(
                provider.extract_pages(document_path),
                {3: "block one\nblock two"},
            )

        self.assertEqual(transport.calls[2]["method"], "GET")
        self.assertEqual(
            transport.calls[2]["url"], "https://example.test/reducto-result.json"
        )

    def test_missing_api_key_raises_clear_error(self):
        transport = FakeTransport([])
        with tempfile.TemporaryDirectory() as temp_dir:
            document_path = Path(temp_dir) / "sample.pdf"
            document_path.write_bytes(b"%PDF-1.4")
            provider = ReductoParseOCRProvider(transport=transport)

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(OCRProviderError, "REDUCTO_API_KEY"):
                    provider.extract_pages(document_path)

    def test_create_ocr_provider_selects_reducto_and_rejects_unknown_names(self):
        provider = create_ocr_provider(
            "reducto-parse", api_key="test-key", transport=FakeTransport([])
        )
        self.assertIsInstance(provider, ReductoParseOCRProvider)
        self.assertIsNone(create_ocr_provider("none"))

        with self.assertRaisesRegex(ValueError, "Unsupported OCR provider"):
            create_ocr_provider("not-real")


if __name__ == "__main__":
    unittest.main()
