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
"""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from review_page.models import InvalidNamedPageKeyReason
from review_page.review_case_messages import convert_message_to_case
from review_page.review_case_source import ReviewCaseSource
from work_matching import read_page_provider, resolve_page_key


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
        mark: Annotated[str, Form()] = "",
        query: Annotated[str, Form()] = "",
        case_name: Annotated[str, Form()] = "",
    ) -> Response:
        """Render the option for a typed page key or address, or nothing for an empty box.

        Args:
            request: The incoming HTTP request.
            page_key: The page whose ``other work`` target is being chosen.
            mark: The page's current mark, kept when the option is chosen.
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
        if read_page_provider(named_entry_page_key) is None:
            invalid_reason: InvalidNamedPageKeyReason | None = (
                InvalidNamedPageKeyReason.NOT_PAGE_KEY
            )
        elif named_entry_page_key in self._list_case_page_keys(case_name):
            invalid_reason = InvalidNamedPageKeyReason.IN_CASE
        else:
            invalid_reason = self._source.find_invalid_named_page_keys(
                [named_entry_page_key]
            ).get(named_entry_page_key)
        return self._templates.TemplateResponse(
            request,
            "named_entry_option.html",
            {
                "page_key": page_key,
                "mark": mark,
                "named_entry_page_key": named_entry_page_key,
                "invalid_reason": invalid_reason,
            },
        )

    def _list_case_page_keys(self, case_name: str) -> set[str]:
        """List the page keys of a case's own entries.

        Args:
            case_name: The case identifier, or empty when none is known.

        Returns:
            Every page key the case's entries hold; none when the case is unknown.
        """
        case_message = self._source.get_case(case_name) if case_name else None
        if case_message is None:
            return set()
        case = convert_message_to_case(case_message)
        return {
            page_key
            for entry in case.entries.values()
            for page_key in entry.addresses_by_page_key
        }
