"""Provide the paths of AniDB's heading tags, the branches of its tag tree that describe no work.

AniDB's anime response lists every ancestor of the tags it applies, so branch names such as
``themes``, ``setting`` or ``maintenance tags`` arrive beside real tags such as ``military``.
AniDB marks no tag as a branch, so the branches are listed here.

A tag is named by its path: its own name after the names of its ancestors, such as
``/setting/time``. A real tag can have children too (``/elements/comedy`` holds
``slapstick``), which is why branches are listed by path rather than found by having children.
"""

HEADING_TAG_PATHS: frozenset[str] = frozenset(
    {
        "/content indicators",
        "/dynamic",
        "/dynamic/cast",
        "/dynamic/ending",
        "/dynamic/storytelling",
        "/elements",
        "/elements/motifs",
        "/elements/pornography",
        "/elements/pornography/group sex",
        "/elements/pornography/oral",
        "/elements/sexual abuse",
        "/elements/speculative fiction",
        "/elements/tropes",
        "/fetishes",
        "/fetishes/breasts",
        "/maintenance tags",
        "/maintenance tags/TO BE MOVED TO CHARACTER",
        "/maintenance tags/TO BE MOVED TO EPISODE",
        "/origin",
        "/original work",
        "/setting",
        "/setting/place",
        "/setting/time",
        "/setting/time/season",
        "/target audience",
        "/technical aspects",
        "/technical aspects/adapted into other media",
        "/technical aspects/awards",
        "/technical aspects/multi-anime projects",
        "/themes",
        "/themes/body and host",
        "/themes/death",
        "/themes/family life",
        "/themes/money",
        "/themes/tales",
        "/ungrouped",
        "/unsorted",
        "/unsorted/character related tags which need deleting or merging",
        "/unsorted/ending tags that need merging",
        "/unsorted/old animetags",
    }
)
