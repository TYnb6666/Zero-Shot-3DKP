"""Slug-to-category mapping for Objaverse / Find3D object identifiers.

Objaverse slugs encode the category as the prefix before the last
underscore, e.g. ``"office_chair_12"`` → ``"office_chair"``.
"""

from __future__ import annotations


def _category_from_slug(slug: str) -> str:
    """Extract the object category from an Objaverse-style slug.

    >>> _category_from_slug("office_chair_12")
    'office_chair'
    >>> _category_from_slug("mug")
    'mug'
    """
    return slug.rsplit('_', 1)[0] if '_' in slug else slug
