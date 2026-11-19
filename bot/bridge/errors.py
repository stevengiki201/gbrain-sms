"""Structured error types shared by the AudioLink service, API and GBrain adapter.

Every failure that crosses a public boundary is represented as an
``AudioLinkError`` with a stable machine-readable ``code``. The API renders it
as ``{"error": {"code": ..., "message": ..., "details": {...}}}``.
"""

from __future__ import annotations

from typing import Any


class AudioLinkError(Exception):
    """Base class for all AudioLink failures."""

    code = "AUDIOLINK_ERROR"
    http_status = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if code:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class ValidationFailed(AudioLinkError):
    code = "VALIDATION_FAILED"
    http_status = 422


class NotFound(AudioLinkError):
    code = "NOT_FOUND"
    http_status = 404


class Conflict(AudioLinkError):
    code = "CONFLICT"
    http_status = 409


class Unauthorized(AudioLinkError):
    code = "UNAUTHORIZED"
    http_status = 401


class Forbidden(AudioLinkError):
    code = "FORBIDDEN"
    http_status = 403


class RateLimited(AudioLinkError):
    code = "RATE_LIMITED"
    http_status = 429


class Expired(AudioLinkError):
    code = "EXPIRED"
    http_status = 410


class Unavailable(AudioLinkError):
    code = "UNAVAILABLE"
    http_status = 503


class MediaUnavailable(AudioLinkError):
    """Raised when an audio operation cannot run on this host.

    Used instead of pretending an audio session exists (e.g. no audio
    transport installed yet, or no sound server / capture device present).
    """

    code = "MEDIA_UNAVAILABLE"
    http_status = 503
