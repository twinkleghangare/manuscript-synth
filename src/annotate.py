"""Ground-truth Markdown writer.

blocks is a list of line-lists in visual reading order:
  [main_lines, main2_lines, note_lines]

Empty inner lists are silently skipped.
"""
from pathlib import Path


def write_md(path, blocks):
    """Write a Markdown ground-truth file for one folio.

    Each non-empty block becomes a paragraph separated by a blank line.
    The marginal note is prefixed with '> ' (blockquote) to distinguish it.
    """
    parts = []
    block_labels = ["main", "main2", "note"]
    for label, block in zip(block_labels, blocks):
        if not block:
            continue
        if label == "note":
            # render marginal annotation as blockquote
            parts.append("\n".join("> " + ln for ln in block))
        else:
            parts.append("\n".join(block))

    Path(path).write_text("\n\n".join(parts) + "\n", encoding="utf-8")
