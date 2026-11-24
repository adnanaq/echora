"""
Match Suggestions - suggest review answers from Jev's judgments of provider pages.

Holds what Jev is asked about two pages of a review case, the decision table
that turns its answers into a relation, the store of its answers, and the
dropdown word each open page of a case is suggested.
"""

python_distribution(
    name="match_suggestions_dist",
    dependencies=[
        "//libs/match_suggestions/src/match_suggestions",
    ],
    provides=python_artifact(
        name="echora-match-suggestions",
        version="0.1.0",
        description="Review answer suggestions from Jev's judgments of provider pages",
    ),
    generate_setup=True,
)
