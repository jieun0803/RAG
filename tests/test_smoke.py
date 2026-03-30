def test_resolve_upstage_api_key_importable():
    from scripts.debug_api_key import resolve_upstage_api_key

    assert callable(resolve_upstage_api_key)
