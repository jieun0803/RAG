import pytest

from scripts.debug_api_key import resolve_upstage_api_key


@pytest.fixture(scope="session")
def upstage_api_key():
    """통합 테스트용: 환경 변수 또는 세션당 1회 getpass."""
    return resolve_upstage_api_key()
