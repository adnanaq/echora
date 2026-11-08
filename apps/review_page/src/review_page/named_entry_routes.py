"""Serve the option the named entry picker offers for a typed page key or address.

The picker's search box filters the suggested entries in the browser. A typed
page key or provider address is checked here, because only the source knows
which keys a seed entry holds and which pages are dead. Each rule reads
``<rule>: <reason>``.

- A valid key comes back as an option that chooses it, so a reviewer can name
  any seed entry, not only the suggested ones.
- An invalid key comes back disabled, with its reason, so a reviewer cannot
  choose a key the save would refuse.
- Text without ``:`` or ``/`` is a search for a suggestion, not a key, so it
  gets no option. Text with one that is not a page key or a provider address
  comes back disabled.
- A key of one of the case's own entries comes back disabled, because such a
  page is placed with ``work N``, as the save's ``NAMED_ENTRY_IN_CASE`` rule
  says.
- A valid key's option shows the seed entry holding it: its title, and its
  reference page when the key is another page of the entry. The reviewer then
  sees which work a pasted address names before choosing it.
- A key whose entry the picker already shows as a suggestion for the typed
  text gets no option, so the entry is offered once. The picker shows a
  suggestion when its title and reference page contain the typed text,
  ignoring case.
"""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from review_page.case_panel_tables import list_target_entry_options
from review_page.models import InvalidNamedPageKeyReason
from review_page.review_case_messages import convert_message_to_case
from review_page.review_case_source import ReviewCaseSource
from work_matching import (
    ReviewCase,
    TargetEntrySuggestion,
    read_page_provider,
    resolve_page_key,
)


class NamedEntryRoutes:
    """Register and serve the named entry picker's typed-key option."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register the typed-key option endpoint on a new router instance.

        Args:
            source: The review case source that tells which page keys are invalid.
            templates: The Jinja2 templates collection for rendering HTML fragments.
        """
        self._source = source
        self._templates = templates
        self.router = APIRouter()
        self.router.add_api_route(
            "/fragments/named-entry-option",
            self.render_named_entry_option,
            methods=["POST"],
        )

    def render_named_entry_option(
        self,
        request: Request,
        page_key: Annotated[str, Form()] = "",
        query: Annotated[str, Form()] = "",
        case_name: Annotated[str, Form()] = "",
    ) -> Response:
        """Render the option for a typed page key or address, or nothing for an empty box.

        Args:
            request: The incoming HTTP request.
            page_key: The page whose ``other work`` target is being chosen.
            query: The text typed in the picker's search box.
            case_name: The case the page belongs to, whose own entries cannot
                be named.

        Returns:
            The option fragment, disabled with its reason when the key cannot
            be used, or an empty response when the text is not a key or an
            address.
        """
        typed_text = query.strip()
        if not any(separator in typed_text for separator in (":", "/")):
            return HTMLResponse("")
        named_entry_page_key = resolve_page_key(typed_text)
        case = self._load_case(case_name)
        invalid_reason = self._find_invalid_reason(named_entry_page_key, case)
        named_entry = (
            None if invalid_reason else self._find_named_entry(named_entry_page_key)
        )
        if named_entry and _is_suggestion_shown(
            named_entry, typed_text, page_key, case
        ):
            return HTMLResponse("")
        return self._templates.TemplateResponse(
            request,
            "named_entry_option.html",
            {
                "page_key": page_key,
                "named_entry_page_key": named_entry_page_key,
                "invalid_reason": invalid_reason,
                "named_entry": named_entry,
            },
        )

    def _load_case(self, case_name: str) -> ReviewCase | None:
        """Load a case by its name.

        Args:
            case_name: The case identifier, or empty when none is known.

        Returns:
            The case, or None when the name is empty or unknown.
        """
        case_message = self._source.get_case(case_name) if case_name else None
        return convert_message_to_case(case_message) if case_message else None

    def _find_invalid_reason(
        self, named_entry_page_key: str, case: ReviewCase | None
    ) -> InvalidNamedPageKeyReason | None:
        """Find why a typed key cannot be named after ``other work``.

        Args:
            named_entry_page_key: The key the typed text resolves to.
            case: The open case, whose own pages cannot be named; None when unknown.

        Returns:
            The reason the key cannot be used, or None when the key can be used.
        """
        if read_page_provider(named_entry_page_key) is None:
            return InvalidNamedPageKeyReason.NOT_PAGE_KEY
        case_page_keys = {
            page_key
            for entry in (case.entries.values() if case else ())
            for page_key in entry.addresses_by_page_key
        }
        if named_entry_page_key in case_page_keys:
            return InvalidNamedPageKeyReason.IN_CASE
        return self._source.find_invalid_named_page_keys([named_entry_page_key]).get(
            named_entry_page_key
        )

    def _find_named_entry(
        self, named_entry_page_key: str
    ) -> TargetEntrySuggestion | None:
        """Find the seed entry holding a typed key.

        Args:
            named_entry_page_key: The key the typed text resolves to.

        Returns:
            The seed entry's reference page, title and providers, or None when
            the source cannot tell, such as without a seed file.
        """
        entry_message = self._source.find_named_entries([named_entry_page_key]).get(
            named_entry_page_key
        )
        if entry_message is None:
            return None
        return TargetEntrySuggestion(
            entry_reference_page=entry_message.entry_reference_page,
            title=entry_message.title,
            providers=tuple(entry_message.providers),
        )


def _is_suggestion_shown(
    named_entry: TargetEntrySuggestion,
    typed_text: str,
    page_key: str,
    case: ReviewCase | None,
) -> bool:
    """Report whether the picker already shows a seed entry as a suggestion for the typed text.

    Args:
        named_entry: The seed entry holding the typed key.
        typed_text: The text typed in the picker's search box.
        page_key: The page whose ``other work`` target is being chosen.
        case: The open case with its suggestions, or None when unknown.

    Returns:
        True when the entry is one of the page's suggestions and its title and
        reference page contain the typed text, ignoring case.
    """
    if case is None:
        return False
    return any(
        suggestion.entry_reference_page == named_entry.entry_reference_page
        and typed_text.lower()
        in f"{suggestion.title} {suggestion.entry_reference_page}".lower()
        for suggestion in list_target_entry_options(case, page_key)
    )
