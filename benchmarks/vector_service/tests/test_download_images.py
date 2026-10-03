from benchmarks.vector_service.test_data.download_images import image_urls


def test_image_urls_json_text_returns_image_urls():
    text = '{"a": "https://cdn.example.com/x/1.jpg", "b": ["https://s4.example.co/y.PNG?t=5"], "c": "no"}'

    assert image_urls(text, excluded_hosts=()) == {
        "https://cdn.example.com/x/1.jpg",
        "https://s4.example.co/y.PNG?t=5",
    }


def test_image_urls_excluded_hosts_are_left_out():
    text = '"https://cdn-eu.anidb.net/images/main/1.jpg" "https://media.kitsu.app/c/2.webp"'

    assert image_urls(text, excluded_hosts=("anidb.net",)) == {
        "https://media.kitsu.app/c/2.webp"
    }


def test_image_urls_escaped_slashes_are_unescaped():
    text = r'"https://cdn.example.com/a.jpg\"'

    assert image_urls(text, excluded_hosts=()) == {"https://cdn.example.com/a.jpg"}


def test_image_urls_only_hosts_keeps_just_those_hosts_even_if_excluded():
    text = '"https://cdn-eu.anidb.net/images/main/1.jpg" "https://media.kitsu.app/c/2.webp"'

    assert image_urls(
        text, excluded_hosts=("anidb.net",), only_hosts=("anidb.net",)
    ) == {"https://cdn-eu.anidb.net/images/main/1.jpg"}
