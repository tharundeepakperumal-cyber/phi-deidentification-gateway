
import re
from typing import Dict


def rehydrate(
    response: str,
    mapping: Dict[str, str]
) -> str:
    """
    Restore only placeholders present in the
    current request's mapping.
    """

    if not mapping:
        return response

    # Match complete placeholders, not partial strings.
    pattern = re.compile(
        "|".join(
            re.escape(key)
            for key in sorted(
                mapping,
                key=len,
                reverse=True
            )
        )
    )

    return pattern.sub(
        lambda match: mapping[match.group()],
        response
    )
