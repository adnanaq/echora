from enrichment.pipeline.page_requests import FETCHED_PROVIDERS, build_page_request


def test_build_page_request_animeschedule_key_gives_animeschedule_request() -> None:
    page_request = build_page_request("animeschedule:anime:one-piece", "One Piece")

    assert page_request is not None
    assert (page_request.service, page_request.id_field, page_request.address) == (
        "animeschedule",
        "animeschedule_url",
        "https://animeschedule.net/anime/one-piece",
    )


def test_build_page_request_provider_without_helper_gives_none() -> None:
    assert build_page_request("ann:anime:836", "One Piece") is None


def test_fetched_providers_include_animeschedule() -> None:
    assert "animeschedule" in FETCHED_PROVIDERS
