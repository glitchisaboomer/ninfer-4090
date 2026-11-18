"""Draft-head shortlist shared with the Qwen3.6 converters."""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.draft_head import (
    DraftHeadContext,
    compute_shortlist as _compute_shortlist,
)

from . import inventory


DEFAULT_RANKING = Path("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64")


def compute_shortlist(
    ranking_path: str | Path,
    tokenizer_dir: str | Path,
) -> DraftHeadContext:
    return _compute_shortlist(
        ranking_path,
        tokenizer_dir,
        n=inventory.DRAFT_HEAD_ROWS,
        vocab=inventory.VOCAB_ROWS,
        tokenizer_vocab_size=inventory.TOKENIZER_DOMAIN,
    )


__all__ = [
    "DEFAULT_RANKING",
    "DraftHeadContext",
    "compute_shortlist",
]
