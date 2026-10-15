import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import app


class PublicDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / "items.csv"
        rows = [
            {"Stock Code": "TA01", "Name": "Copper cable", "Stock Group": "TA",
             "Rated voltage": "450 / 750V", "Data sheet": "https://example.com/ta.pdf",
             "Create User Name": "privatestaffmarker", "Nexans": "privatecrossrefmarker",
             "Future Internal Column": "privatefuturemarker"},
            {"Stock Code": "TA02", "Name": "Rubber cable", "Stock Group": "TA",
             "Rated voltage": "0.6/1kV", "Data sheet": "https://example.com/tb.pdf",
             "Create User Name": "anotherstaffmarker", "Nexans": "anothercrossrefmarker",
             "Future Internal Column": "anotherfuturemarker"},
        ]
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        df, idf = app.load_items(str(self.csv_path), None)
        state = patch.multiple(app, _df=df, _idf=idf, _data_load_error="",
                               API_KEY="", ALLOW_PUBLIC_API=True)
        state.start()
        self.addCleanup(state.stop)
        self.client = app.app.test_client()
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_internal_fields_are_neither_returned_nor_searchable(self):
        for marker in ("privatestaffmarker", "privatecrossrefmarker", "privatefuturemarker"):
            self.assertNotIn(marker, app._idf)
            self.assertFalse(app._df["__search_blob"].str.contains(marker).any())
            response = self.client.get("/items/filter", query_string={"include": marker})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["results"], [])
        item = app.df_row_to_item({"Stock Code": "TA01", "Future Internal Column": "secret"})
        self.assertEqual(item, {"Stock Code": "TA01"})

    def test_product_routes_and_nested_responses_use_public_fields(self):
        cases = [
            ("GET", "/items/by_code?q=TA01", None, 200),
            ("GET", "/items/by_code?q=TA03", None, 404),
            ("GET", "/items/search?q=cable", None, 200),
            ("GET", "/items/filter?include=cable", None, 200),
            ("POST", "/items/filter", {"include": ["cable"]}, 200),
            ("POST", "/ask", {"query": "cable"}, 200),
            ("POST", "/router/ask", {"query": "What is TA01"}, 200),
            ("POST", "/router/ask", {"query": "TA01 voltage"}, 200),
            ("POST", "/router/ask", {"query": "Compare difference TA01 vs TA02"}, 200),
            ("POST", "/router/ask", {"query": "cable"}, 200),
        ]
        for method, url, body, status in cases:
            with self.subTest(url=url, body=body):
                response = self.client.open(url, method=method, json=body)
                self.assertEqual(response.status_code, status)
                text = response.get_data(as_text=True)
                for forbidden in ("Create User Name", "Nexans", "Future Internal Column",
                                  "privatestaffmarker", "privatecrossrefmarker", "privatefuturemarker"):
                    self.assertNotIn(forbidden, text)
                self.assertIn("TA01", text)
        item = self.client.get("/items/by_code?q=TA01").json
        self.assertEqual(item["Rated voltage"], "450 / 750V")
        self.assertEqual(item["Data sheet"], "https://example.com/ta.pdf")
        comparison = self.client.post("/router/ask", json={"query": "Compare difference TA01 vs TA02"}).json
        self.assertEqual(comparison["source"], "compare")
        self.assertEqual(len(comparison["comparison"]["items"]), 2)

    def test_auth_modes_cover_every_product_route(self):
        routes = [("GET", "/items/by_code?q=TA01", None),
                  ("GET", "/items/search?q=cable", None),
                  ("GET", "/items/filter", None),
                  ("POST", "/items/filter", {}),
                  ("POST", "/ask", {"query": "cable"}),
                  ("POST", "/router/ask", {"query": "TA01"})]
        for key, public, supplied, expected in [
            ("", False, "", 401), ("", False, "anything", 401),
            ("", True, "", 200), ("test-only-key", False, "", 401),
            ("test-only-key", False, "wrong", 401),
            ("test-only-key", True, "", 401),
            ("test-only-key", True, "test-only-key", 200),
            ("test-only-key", False, "test-only-key", 200),
        ]:
            with patch.multiple(app, API_KEY=key, ALLOW_PUBLIC_API=public):
                for method, url, body in routes:
                    with self.subTest(key_set=bool(key), public=public, supplied=supplied, url=url):
                        response = self.client.open(url, method=method, json=body,
                                                    headers={"X-API-Key": supplied})
                        self.assertEqual(response.status_code, expected)

    def test_missing_environment_does_not_enable_public_access(self):
        env = {k: v for k, v in os.environ.items() if k not in ("API_KEY", "ALLOW_PUBLIC_API")}
        env["ITEMS_CSV_PATH"] = str(self.csv_path)
        env["SERIES_CSV_PATH"] = str(Path(self.tmp.name) / "missing.csv")
        result = subprocess.run([sys.executable, "-c", "import app; assert not app.API_KEY; "
                                 "assert not app.ALLOW_PUBLIC_API; "
                                 "assert app.app.test_client().get('/items/by_code?q=TA01').status_code == 401"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_health_check_does_not_disclose_source_paths(self):
        with patch.object(app, "_data_load_error", "/private/export/customer.csv: bad column"):
            response = self.client.get("/healthz")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json, {"ok": False, "error": "product data unavailable"})


if __name__ == "__main__":
    unittest.main()
