"""Hold the answer a case takes over from the case it replaces after its pages changed."""

from pydantic import BaseModel, ConfigDict, JsonValue


class PreviousAnswer(BaseModel):
    """Hold an answered case's answer, carried to the case that replaces it.

    A seed release adding a page to an entry renames the entry's case, so the
    case's answer no longer meets it by name. ``previous_answers`` finds the
    answered case the new case replaces and records its answer here.
    """

    model_config = ConfigDict(frozen=True)

    # The name of the answered case this case replaces.
    case_name: str
    # The answer as written for that case.
    answer: JsonValue
    # The pages this case has that the previous case did not, sorted.
    added_page_keys: tuple[str, ...] = ()
    # Whether the previous answer still places this case's pages, which holds when the
    # case has the same entries as the previous case and the answer was valid
    # there.
    is_applied: bool
