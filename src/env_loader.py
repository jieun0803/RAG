"""config.env 스타일 파일 파싱 및 UPSTAGE_API_KEY 검증."""

from __future__ import annotations

import os
from pathlib import Path

UPSTAGE_KEY = "UPSTAGE_API_KEY"
PLACEHOLDER = "your_upstage_api_key_here"


class EnvironmentLoader:
    """KEY=VALUE 줄 단위 파싱. # 주석·빈 줄 무시."""

    @staticmethod
    def parse_file(path: str | Path) -> dict[str, str]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        out: dict[str, str] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key:
                out[key] = value
        return out


def load_environment(env_file: str | Path = "config.env", *, strict: bool = True) -> None:
    """파일이 있으면 로드해 os.environ에 반영(이미 설정된 키는 덮어쓰지 않음). strict면 UPSTAGE_API_KEY 검증."""
    path = Path(env_file)
    if path.is_file():
        for k, v in EnvironmentLoader.parse_file(path).items():
            os.environ.setdefault(k, v)
    if strict:
        get_api_key()


def get_api_key() -> str:
    key = (os.environ.get(UPSTAGE_KEY) or "").strip()
    if not key:
        raise ValueError(
            f"{UPSTAGE_KEY} is not set. Use load_environment(), set the env var, "
            "or set_api_key() before calling the API."
        )
    if key == PLACEHOLDER:
        raise ValueError(
            f"{UPSTAGE_KEY} is still the placeholder; replace it with a real key."
        )
    return key


def set_api_key(value: str) -> None:
    """Streamlit UI 등에서 받은 키를 os.environ에 넣기. 플레이스홀더·빈 값은 거부."""
    v = (value or "").strip()
    if not v:
        raise ValueError(f"{UPSTAGE_KEY} cannot be empty.")
    if v == PLACEHOLDER:
        raise ValueError(f"{UPSTAGE_KEY} cannot be the placeholder value.")
    os.environ[UPSTAGE_KEY] = v
