"""Magic-byte file type verification (defense-in-depth vs bad uploads)."""
from __future__ import annotations

import filetype

PDF_MAGIC = b"%PDF-"


def is_pdf(head: bytes) -> bool:
    if head.startswith(PDF_MAGIC):
        return True
    kind = filetype.guess(head)
    return bool(kind and kind.mime == "application/pdf")


class InvalidFileType(Exception):
    ...


def assert_pdf(head: bytes) -> None:
    if not is_pdf(head):
        raise InvalidFileType("Uploaded file is not a valid PDF (magic-byte check failed).")
