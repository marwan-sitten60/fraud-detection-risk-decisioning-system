"""Sequential HTTP adapter with bounded idempotent retries."""

import time

import httpx2 as httpx


class SimulationHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SimulationHTTPClient:
    def __init__(self, base_url: str, timeout_seconds: float, retry_limit: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.retry_limit = retry_limit
        self.client = httpx.Client(timeout=timeout_seconds)
        self.retry_count = 0

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, *, retries: bool = False, **kwargs):
        attempts = self.retry_limit if retries else 0
        for attempt in range(attempts + 1):
            try:
                response = self.client.request(method, self.base_url + path, **kwargs)
            except httpx.TransportError as error:
                if attempt >= attempts:
                    raise SimulationHTTPError(f"HTTP transport failed for {path}") from error
                self._backoff(attempt)
                continue
            if response.status_code == 409:
                raise SimulationHTTPError(f"HTTP 409 conflict for {path}", 409)
            if (response.status_code >= 500 or response.status_code == 429) and attempt < attempts:
                self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise SimulationHTTPError(
                    f"HTTP {response.status_code} for {path}: {response.text[:300]}",
                    response.status_code,
                )
            return response
        raise AssertionError("unreachable retry state")

    def _backoff(self, attempt: int) -> None:
        self.retry_count += 1
        time.sleep(min(0.25 * (2**attempt), 1.0))
