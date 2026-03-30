"""백엔드·터미널 디버그용: 환경 변수에 키가 없으면 한 번만 물어봅니다."""

from __future__ import annotations

import os
from getpass import getpass

_DEFAULT_ENV = "UPSTAGE_API_KEY"


def resolve_upstage_api_key(env_name: str = _DEFAULT_ENV) -> str:
    key = (os.environ.get(env_name) or "").strip()
    if key:
        return key
    key = getpass(f"{env_name} is not set. Enter API key (hidden): ").strip()
    if not key:
        raise ValueError(
            f"{env_name} is required. Set the environment variable or enter when prompted."
        )
    os.environ[env_name] = key
    return key


if __name__ == "__main__":
    resolved = resolve_upstage_api_key()
    masked = f"{resolved[:4]}…{resolved[-4:]}" if len(resolved) > 8 else "(set)"
    print(f"Resolved {_DEFAULT_ENV}: {masked}")
