from __future__ import annotations

from typing import Any, Dict, Optional


def api_success(data: Any, diagnostics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "error": None,
        "diagnostics": diagnostics or {},
    }


def api_error(
    code: str,
    message: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
        },
        "diagnostics": diagnostics or {},
    }
