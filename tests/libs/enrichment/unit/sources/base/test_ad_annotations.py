from enrichment.sources.base.ad_annotations import remove_ad_annotations


def _annotated(word: str) -> str:
    return (
        '<a href="#" class="google-anno" data-google-vignette="false" '
        'data-google-interstitial="false" style="position: initial !important;">'
        '<svg viewBox="100 -1000 840 840" width="calc(22.4px - 2px)" height="22.4px">'
        '<path d="M168-144q-29.7 0-50.85-21.15Z"></path></svg>&nbsp;'
        f'<span class="google-anno-t" style="text-decoration: underline dotted !important;">{word}</span></a>'
    )


def test_the_annotated_word_is_put_back() -> None:
    page = f"<h3>{_annotated('Anime')} Roles</h3>"
    assert remove_ad_annotations(page) == "<h3>Anime Roles</h3>"


def test_a_page_without_annotations_is_unchanged() -> None:
    page = '<h3>Anime Roles</h3><table><tr><td><a href="/anime/one-piece">One Piece</a></td></tr></table>'
    assert remove_ad_annotations(page) == page


def test_every_annotation_on_the_page_is_removed() -> None:
    page = (
        f"<h3>{_annotated('Anime')} Roles</h3>"
        f'<table><tr><td><a href="/anime/one-piece">One Piece {_annotated("Film")}</a></td></tr></table>'
    )
    assert remove_ad_annotations(page) == (
        "<h3>Anime Roles</h3>"
        '<table><tr><td><a href="/anime/one-piece">One Piece Film</a></td></tr></table>'
    )


def test_an_annotation_without_its_word_span_keeps_its_visible_text() -> None:
    page = '<h3><a href="#" class="google-anno"><svg></svg>&nbsp;Anime</a> Roles</h3>'
    assert remove_ad_annotations(page) == "<h3>Anime Roles</h3>"


def _suggestion_chip(first: str, rest: str) -> str:
    return (
        '<div class="google-anno-skip google-anno-sc" tabindex="0" role="link" '
        f'aria-label="{first} {rest}" data-google-vignette="false" '
        'data-google-interstitial="false" style="display: inline-flex !important;">'
        '<span><span><svg viewBox="0 -960 960 960" width="16px" height="16px">'
        '<path d="M168-144q-29.7 0-50.85-21.15Z"></path></svg></span>'
        f"<span>{first}</span></span><span>{rest}</span></div>"
    )


def test_a_suggestion_chip_inside_the_text_is_removed() -> None:
    page = (
        '<div itemprop="description"><p>The Grand Line. '
        f"{_suggestion_chip('Anime', '&amp; Manga')}</p></div>"
    )
    assert remove_ad_annotations(page) == (
        '<div itemprop="description"><p>The Grand Line. </p></div>'
    )


def test_links_and_chips_on_one_page_are_both_undone() -> None:
    page = (
        f"<h3>{_annotated('Anime')} Roles</h3>"
        f"<p>Read on. {_suggestion_chip('Read', 'Popular Manga')}</p>"
    )
    assert remove_ad_annotations(page) == "<h3>Anime Roles</h3><p>Read on. </p>"


def test_widgets_outside_the_page_content_are_left_alone() -> None:
    page = (
        '<div class="goog-rentries"><div class="google-anno-skip goog-rentry">x</div></div>'
        '<div id="google-anno-sa" dir="ltr"><div id="google-anno-sa-glow"></div></div>'
    )
    assert remove_ad_annotations(page) == page
