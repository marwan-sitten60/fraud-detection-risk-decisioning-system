"""Small read-only HTTP client for the existing FastAPI service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ApiResult:
    data: dict | list | None
    error: str | None = None
    status_code: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class APIClient:
    """Read-only, timeout-bounded adapter. It never reads database settings."""

    def __init__(self, base_url: str | None = None, timeout: float = 5.0) -> None:
        self.base_url = base_url or os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8000")
        self.base_url = self.base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: dict | None = None) -> ApiResult:
        query = urlencode(
            {key: value for key, value in (params or {}).items() if value is not None}
        )
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = response.status
                raw = response.read()
        except HTTPError as error:
            return ApiResult(None, f"API returned HTTP {error.code}", error.code)
        except (TimeoutError, URLError, OSError):
            return ApiResult(None, "Could not connect to the fraud API")
        if not 200 <= status < 300:
            return ApiResult(None, f"API returned HTTP {status}", status)
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ApiResult(None, "API returned invalid JSON", status)
        if not isinstance(parsed, (dict, list)):
            return ApiResult(None, "API returned an unexpected response", status)
        return ApiResult(parsed, status_code=status)
