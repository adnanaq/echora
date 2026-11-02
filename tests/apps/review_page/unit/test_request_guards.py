import json


def _save_form(review_cases) -> dict[str, str]:
    case = review_cases["atashinchi_next"]
    return {
        "browser_state": json.dumps({"edits": {case.case_name: case.suggested_answer}}),
        "action": "save",
        "case_name": case.case_name,
    }


def test_add_request_guards_foreign_host_answers_400(client) -> None:
    response = client.get("/health", headers={"Host": "attacker.example:8003"})
    assert response.status_code == 400


def test_add_request_guards_localhost_name_answers_ok(client) -> None:
    response = client.get("/health", headers={"Host": "localhost:8003"})
    assert response.status_code == 200


def test_add_request_guards_save_without_htmx_header_answers_403_and_saves_nothing(
    client, source, review_cases
) -> None:
    response = client.post(
        "/answers", headers={"HX-Request": "false"}, data=_save_form(review_cases)
    )
    case_name = review_cases["atashinchi_next"].case_name
    assert (response.status_code, source.cases[case_name].answer) == (403, None)


def test_add_request_guards_run_without_htmx_header_answers_403_and_starts_nothing(
    client, source
) -> None:
    response = client.post("/run", headers={"HX-Request": "false"}, data={})
    assert (response.status_code, source.is_running) == (403, False)


def test_add_request_guards_page_without_htmx_header_answers_ok(client) -> None:
    response = client.get("/", headers={"HX-Request": "false"})
    assert response.status_code == 200


def test_add_request_guards_correction_fragment_without_htmx_header_answers_403(
    client,
) -> None:
    response = client.post(
        "/fragments/correction-list", headers={"HX-Request": "false"}, data={}
    )
    assert response.status_code == 403


def test_add_request_guards_static_file_with_foreign_host_answers_400(client) -> None:
    response = client.get(
        "/static/parts.css", headers={"Host": "attacker.example:8003"}
    )
    assert response.status_code == 400


def test_add_request_guards_preflight_from_other_origin_gets_no_cors_permission(
    client,
) -> None:
    response = client.options(
        "/answers",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "hx-request",
        },
    )
    assert "access-control-allow-origin" not in response.headers
