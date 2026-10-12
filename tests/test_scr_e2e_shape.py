"""Host-side checks for the SCR e2e script's fixtures and plan-shape assertions.

The interleaved fixture exists to settle one question that no run has answered: whether segmented
relocation does anything tail relocation does not. Both selectors return the same span on every
prompt built before this shape, so both modes measured identically and the interleaved claim was
reasoned rather than observed. These tests prove the *shape* of the fixture on any machine -- two
deletions leaving two separated survivors -- using ports of the detector and the selectors that
`src/targets/qwen3_6/impl/runtime/prompt_edit.{h,cpp}` implements, driven by the script's own
alignment search.

No server, no GPU, no tokenizer assets: the alignment search is checked against a deterministic
stand-in that counts words, and the planner arithmetic against token sequences built from it.
"""

from __future__ import annotations

import contextlib

import pytest

from tools import scr_e2e_check

BLOCK = scr_e2e_check.BLOCK_TOKENS

# The chat template wraps every prompt, and the turn-rewrite boundary sits between the user
# content and the assistant header. The stand-in models exactly that: a fixed prefix, the content,
# a fixed suffix.
TEMPLATE_PREFIX_TOKENS = 4
TEMPLATE_SUFFIX_TOKENS = 12


def tokenize(text, vocab):
    """One token per word, with a fresh id per distinct word.

    Enough to make the block diff real: identical text yields identical token ids, and any
    difference yields different ones. The word count is what the alignment search measures.
    """
    return [vocab.setdefault(word, len(vocab)) for word in text.split()]


def server_tokens(text, vocab):
    """The token id sequence the server would prefill for `text`, template included."""
    return ([-1] * TEMPLATE_PREFIX_TOKENS + tokenize(text, vocab) +
            [-2] * TEMPLATE_SUFFIX_TOKENS)


def count_words(port, prompt, max_tokens, timeout, think):
    """Stand-in for the server's own tokenizer, reached through /v1/messages/count_tokens."""
    return TEMPLATE_PREFIX_TOKENS + len(prompt.split()) + TEMPLATE_SUFFIX_TOKENS


def turn_boundary(text):
    """Where the assistant's turn begins: the end of the user content, in tokens."""
    return TEMPLATE_PREFIX_TOKENS + len(text.split())


@contextlib.contextmanager
def live_shape(shape):
    """Set the script's live fixture shape and put the shipped default back afterwards."""
    previous = scr_e2e_check.SHAPE
    scr_e2e_check.SHAPE = shape
    try:
        yield shape
    finally:
        scr_e2e_check.SHAPE = previous


@pytest.fixture(autouse=True)
def word_tokenizer(monkeypatch):
    """Every test runs under the stand-in tokenizer, restored afterwards."""
    monkeypatch.setattr(scr_e2e_check, "count_tokens", count_words)


# ---------------------------------------------------------------------------------------------
# Ports of the planner this fixture is aimed at. They are held here rather than imported because
# they are the oracle: what the fixture does is decided by these rules, so the test has to carry
# its own copy of them or it is only checking the script against itself.
# ---------------------------------------------------------------------------------------------
def surviving_spans(previous_tokens, current_tokens, block=BLOCK):
    """A port of detail::diff_prompt_edit: whole blocks matched by content, monotone, disjoint."""
    index = {}
    for b in range(len(previous_tokens) // block):
        index.setdefault(tuple(previous_tokens[b * block:(b + 1) * block]), []).append(b)
    spans = []
    old_cursor = 0
    new_block = 0
    while new_block < len(current_tokens) // block:
        key = tuple(current_tokens[new_block * block:(new_block + 1) * block])
        matched = False
        for old_block in index.get(key, []):
            if old_block < old_cursor:  # a block is never reused
                continue
            run = 1
            while (old_block + run < len(previous_tokens) // block and
                   new_block + run < len(current_tokens) // block and
                   previous_tokens[(old_block + run) * block:(old_block + run + 1) * block] ==
                   current_tokens[(new_block + run) * block:(new_block + run + 1) * block]):
                run += 1
            spans.append((old_block * block, new_block * block, run * block))
            old_cursor = old_block + run
            new_block += run
            matched = True
            break
        if not matched:
            new_block += 1
    return spans


def candidates(spans, resident_tokens, limit):
    """A port of the candidate rules both selectors share (plan_relocations, lines 68-97)."""
    out = []
    for old_begin, new_begin, tokens in spans:
        if old_begin == 0 and new_begin == 0:
            continue  # the leading span is ordinary prefix reuse
        if old_begin + tokens > resident_tokens or tokens == 0 or new_begin >= limit:
            continue
        fit = ((limit - new_begin) // BLOCK) * BLOCK
        if fit == 0:
            continue
        out.append((new_begin, min(tokens, fit)))
    return sorted(out)


def select_segmented(spans, resident_tokens, limit, max_spans):
    """A port of detail::plan_relocations' selection: the `max_spans` longest survivors.

    The cap is applied longest-first and the survivors are then handed back in prompt order, which
    is the order the runtime consumes (`plan_relocations` sorts by `dest_begin` after the cap).
    """
    return sorted(sorted(candidates(spans, resident_tokens, limit),
                         key=lambda span: -span[1])[:max_spans])


def select_tail(spans, resident_tokens, limit, max_spans):
    """A port of detail::plan_tail_relocation: the longest contiguous run, windowed to the cap.

    The two selectors share the candidate rules above; the only difference is that tail takes one
    window out of a run of adjacent survivors, and a gap between survivors breaks the run.
    """
    pool = candidates(spans, resident_tokens, limit)
    if not pool:
        return []
    cap = min(max_spans, len(pool))
    best, best_total = [], 0
    run_begin = 0
    while run_begin < len(pool):
        run_end = run_begin + 1
        while (run_end < len(pool) and
               pool[run_end][0] == pool[run_end - 1][0] + pool[run_end - 1][1]):
            run_end += 1
        run_length = run_end - run_begin
        window = min(cap, run_length)
        total = sum(t for _d, t in pool[run_begin:run_begin + window])
        best_sum, best_offset = total, 0
        for offset in range(1, run_length - window + 1):
            total = (total - pool[run_begin + offset - 1][1] +
                     pool[run_begin + offset + window - 1][1])
            if total > best_sum:
                best_sum, best_offset = total, offset
        if best_sum > best_total:
            best_total = best_sum
            best = pool[run_begin + best_offset:run_begin + best_offset + window]
        run_begin = run_end
    return best


def prefilled_ranges(selected, prompt_tokens, prefix_reuse=0):
    """A port of detail::relocation_prefill_ranges: every suffix token no span covers."""
    ranges = []
    cursor = prefix_reuse
    for dest_begin, tokens in selected:
        if dest_begin > cursor:
            ranges.append((cursor, dest_begin))
        cursor = max(cursor, dest_begin + tokens)
    if cursor < prompt_tokens:
        ranges.append((cursor, prompt_tokens))
    return ranges


class Geometry:
    """One fixture, once aligned: its spans, its resident frontier, and its prompt length."""

    def __init__(self, shape):
        # Both the alignment search and the prompts it aligned must be built in `shape`:
        # `request_prompt` reads the module-level SHAPE, and `resolve_alignment` restores it when it
        # returns, so building the prompts outside this window silently falls back to Scope A and the
        # geometry then describes a different fixture than the one under test.
        with live_shape(shape):
            pads = resolve_alignment(shape)
            assert pads is not None, "the alignment search failed"
            vocab = {}
            self.text_a = scr_e2e_check.request_prompt("a", pads)
            self.text_b = scr_e2e_check.request_prompt("b", pads)
            self.prompt_a = server_tokens(self.text_a, vocab)
            self.prompt_b = server_tokens(self.text_b, vocab)
            self.resident = len(self.prompt_a)
            self.limit = min(len(self.prompt_b) - 1, turn_boundary(self.text_b))
            self.spans = surviving_spans(self.prompt_a, self.prompt_b)

    def segmented(self, max_spans):
        return select_segmented(self.spans, self.resident, self.limit, max_spans)

    def tail(self, max_spans):
        return select_tail(self.spans, self.resident, self.limit, max_spans)


def resolve_alignment(shape):
    """Run the script's own alignment search under the stand-in tokenizer, in `shape`."""
    with live_shape(shape):
        return scr_e2e_check.solve_alignment(None, 512, 900, False)


# ---------------------------------------------------------------------------------------------
# The alignment search
# ---------------------------------------------------------------------------------------------
def test_interleaved_alignment_aligns_every_boundary() -> None:
    with live_shape("interleaved"):
        pads = scr_e2e_check.solve_alignment(None, 512, 900, False)

    assert pads is not None
    assert len(pads) == 5, "the interleaved shape has five regions to align"
    assert all(len(pad) == 2 for pad in pads), "every pad is a (units, shift) pair"
    with live_shape("interleaved"):
        for (label, _why, build), pad in zip(scr_e2e_check._alignment_targets(pads), pads):
            length = count_words(None, build(pad), 512, 900, False)
            assert length % BLOCK == 0, f"{label} ends at {length} tokens, not on a block boundary"


def test_scope_a_alignment_aligns_its_two_boundaries() -> None:
    with live_shape("scope-a"):
        pads = scr_e2e_check.solve_alignment(None, 512, 900, False)
        targets = scr_e2e_check._alignment_targets(pads)

    assert pads is not None
    assert len(targets) == 2, "Scope A has one survivor, hence two boundaries"
    with live_shape("scope-a"):
        for (_label, _why, build), pad in zip(targets, pads):
            assert count_words(None, build(pad), 512, 900, False) % BLOCK == 0
    # The three interleaved regions are absent from this shape, so they stay unpadded.
    assert pads[2:] == [(0, 0), (0, 0), (0, 0)]


def test_the_alignment_shift_reaches_boundaries_an_even_pad_cannot(monkeypatch) -> None:
    """A pad whose token count is even shares a factor with 64 and reaches only half the
    boundaries, so a walk over whole pad units can fail for a reason that has nothing to do with
    relocation. The one-word shift is what makes every boundary reachable whatever the filler's
    own count turns out to be, and this is the test that it does."""
    def build(pad):
        units, shift = pad
        return "u" * units + "s" * shift

    # Every unit costs 14 tokens (even), and every shift word costs 1 (odd). The base is odd, so
    # no number of whole units can ever land on a boundary; only the shift dimension can.
    def count(port, text, max_tokens, timeout, think):
        return 17 + 14 * text.count("u") + text.count("s")

    monkeypatch.setattr(scr_e2e_check, "count_tokens", count)
    monkeypatch.setattr(scr_e2e_check, "ALIGNMENT_SHIFT_SPAN", 1)
    assert scr_e2e_check._search_alignment(None, "TEST", "why", build, 512, 900, False) is None, \
        "a shift width of one must be unable to align this pad"

    monkeypatch.setattr(scr_e2e_check, "ALIGNMENT_SHIFT_SPAN", 4)
    pad = scr_e2e_check._search_alignment(None, "TEST", "why", build, 512, 900, False)
    assert pad is not None, "the shift dimension was not enough to reach a boundary"
    units, shift = pad
    assert shift > 0, "the alignment was found without using the shift at all"
    assert count(None, build(pad), 512, 900, False) % BLOCK == 0


# ---------------------------------------------------------------------------------------------
# The interleaved fixture's shape -- the whole point of it
# ---------------------------------------------------------------------------------------------
def test_interleaved_fixture_leaves_two_separated_survivors() -> None:
    geometry = Geometry("interleaved")

    # Three spans: the shared prefix, the middle survivor, the tail survivor.
    assert len(geometry.spans) == 3, f"expected P, S1 and S2, got {geometry.spans}"
    (p_old, p_new, p_len), (s1_old, s1_new, s1_len), (s2_old, s2_new, s2_len) = geometry.spans
    assert p_old == 0 and p_new == 0, "the leading span is the shared prefix"
    assert p_len != 0, "the shared prefix was not matched at all"
    assert s1_old + s1_len < s2_old, "S1 and S2 are separated in prompt A"
    assert s1_new + s1_len < s2_new, "S1 and S2 are separated in prompt B"
    assert s1_new + s1_len + BLOCK <= s2_new, "there is a whole block of gap between them"
    assert s2_new + s2_len <= geometry.limit, "S2 stops short of the rewrite boundary"

    # Segmented relocates both survivors, and derives one prefilled range per gap.
    selected = geometry.segmented(6)
    assert len(selected) == 2, f"segmented relocated {len(selected)} spans, expected both"
    assert [dest for dest, _tokens in selected] == [s1_new, s2_new]
    ranges = prefilled_ranges(selected, len(geometry.prompt_b))
    assert len(ranges) == 3, f"expected three ranges, got {ranges}"
    assert ranges[0][0] == 0, \
        "an in-the-middle edit reuses no prefix, so the first range starts at 0"
    assert ranges[-1][1] == len(geometry.prompt_b), "the final prompt token is always prefilled"

    # The planner's own tiling contract: a token is prefilled or relocated, never unaccounted.
    tiled = sum(end - begin for begin, end in ranges) + sum(t for _d, t in selected)
    assert tiled == len(geometry.prompt_b)


def test_tail_mode_relocates_one_survivor_where_segmented_relocates_both() -> None:
    """The distinction the whole fixture exists to measure, on one prompt."""
    geometry = Geometry("interleaved")
    segmented = geometry.segmented(6)
    tail = geometry.tail(6)

    assert len(segmented) == 2
    assert len(tail) == 1, "the tail planner relocates a single run, and the survivors are two"
    assert 0 < sum(t for _d, t in tail) < sum(t for _d, t in segmented), \
        "segmented must relocate strictly more than tail on this fixture"


def test_the_span_cap_collapses_segmented_onto_one_survivor() -> None:
    geometry = Geometry("interleaved")
    both = geometry.segmented(6)
    one = geometry.segmented(1)
    assert len(one) == 1 and len(both) == 2
    assert one == [max(both, key=lambda span: span[1])], "the cap keeps the longer survivor"
    assert sum(t for _d, t in one) < sum(t for _d, t in both)

    assert geometry.segmented(0) == [], "a zero cap relocates nothing: the flag-phase control"


def test_scope_a_produces_one_survivor_however_it_is_selected() -> None:
    """The built-in suite's blind spot, asserted so it cannot be mistaken for a passing run: one
    surviving run means both selectors choose the same span, so neither mode can be distinguished
    on it."""
    geometry = Geometry("scope-a")
    assert len(geometry.spans) == 2, "Scope A's edit leaves the prefix and one surviving run"
    segmented = geometry.segmented(6)
    tail = geometry.tail(6)
    assert segmented == tail, "the two selectors agree on Scope A, which is why it cannot tell \
them apart"
    assert len(segmented) == 1
    ranges = prefilled_ranges(segmented, len(geometry.prompt_b))
    tiled = sum(end - begin for begin, end in ranges) + sum(t for _d, t in segmented)
    assert tiled == len(geometry.prompt_b)


def test_padding_one_region_leaves_the_others_alone() -> None:
    """A pad must move only the region it pads, or the searches are not independent and one pad
    can silently shift another region's boundary."""
    with live_shape("interleaved"):
        base = scr_e2e_check.request_prompt("a", [(0, 0)] * 5)
        tail_unpadded = scr_e2e_check.tail_text()
        first_plain = scr_e2e_check.first_removed_text((0, 0))

        padded = scr_e2e_check.request_prompt("a", [(0, 0), (2, 1), (0, 0), (0, 0), (0, 0)])
        again = scr_e2e_check.request_prompt("a", [(0, 0), (2, 1), (0, 0), (0, 0), (0, 0)])

    assert padded == again, "the same pads must produce the same prompt, or runs are not repeatable"
    assert padded != base
    assert scr_e2e_check.tail_text() == tail_unpadded, "the tail survivor is never padded"
    assert padded.endswith(tail_unpadded.strip()), "the tail still ends the prompt"
    # Only the first removed region grew: everything before P and after R1 is untouched.
    assert padded.startswith(scr_e2e_check.prefix_text((0, 0)))
    assert padded.count(scr_e2e_check.FIRST_PAD + " " + scr_e2e_check.FIRST_PAD) == 1
    assert len(padded) - len(base) == len(scr_e2e_check.first_removed_text((2, 1))) - \
        len(first_plain)


# ---------------------------------------------------------------------------------------------
# The plan-shape assertions read out of `--scr-diag`
# ---------------------------------------------------------------------------------------------
def well_formed_plan():
    """Two separated survivors, one prefilled range per gap, exactly tiled."""
    return {
        "prompt": 1000,
        "base": 0,
        "spans": [{"src": 832, "dst": 640, "tok": 128}, {"src": 960, "dst": 832, "tok": 128}],
        "ranges": [(0, 640), (768, 832), (960, 1000)],
        "ranges_reported": 3,
    }


def test_well_formed_plan_has_no_findings() -> None:
    assert scr_e2e_check.check_relocation_plan(well_formed_plan(), 2) == []


def test_plan_shape_checks_catch_a_mis_tiled_plan() -> None:
    plan = well_formed_plan()
    plan["ranges"] = [(0, 576), (768, 832), (960, 1000)]  # a gap: [576,640) unaccounted
    problems = scr_e2e_check.check_relocation_plan(plan, 2)
    assert any("unaccounted" in problem for problem in problems)


def test_plan_shape_checks_catch_an_overlapping_range() -> None:
    plan = well_formed_plan()
    plan["ranges"] = [(0, 700), (672, 832), (960, 1000)]
    problems = scr_e2e_check.check_relocation_plan(plan, 2)
    assert any("overlaps" in problem for problem in problems)


def test_plan_shape_checks_catch_the_wrong_number_of_survivors() -> None:
    assert scr_e2e_check.check_relocation_plan(well_formed_plan(), 1), \
        "two spans where one was claimed"
    plan = well_formed_plan()
    plan["spans"] = plan["spans"][:1]
    plan["ranges"] = [(0, 640), (768, 1000)]
    assert scr_e2e_check.check_relocation_plan(plan, 2), "one span where two were claimed"


def test_plan_shape_checks_catch_a_span_that_reaches_the_final_token() -> None:
    plan = well_formed_plan()
    plan["spans"][1] = {"src": 960, "dst": 832, "tok": 192}
    plan["ranges"] = [(0, 640), (768, 832)]
    problems = scr_e2e_check.check_relocation_plan(plan, 2)
    assert any("final prompt token" in problem for problem in problems)


def test_plan_shape_checks_catch_a_span_that_is_not_whole_blocks() -> None:
    plan = well_formed_plan()
    plan["spans"][0] = {"src": 800, "dst": 641, "tok": 128}
    problems = scr_e2e_check.check_relocation_plan(plan, 2)
    assert any("64-token blocks" in problem for problem in problems)


def test_a_plan_line_parses_back_off_the_diagnostic() -> None:
    line = ("[scr-diag] prompt_tokens=1000 base=0 rope_freq[0]=1e-06 rope_freq[17]=1e-05 "
            "span{src=832 dst=640 tok=128 delta=-192 src_pages=13..14 dst_pages=10..11} "
            "span{src=960 dst=832 tok=128 delta=-128 src_pages=15..16 dst_pages=13..14} "
            "ranges=3 [0,640) [768,832) [960,1000) remap[-1,5,-1,6]")
    plans = scr_e2e_check.parse_relocation_plans(
        "[scr-diag] declined prompt_tokens=1 reuse_base=0\n" + line)
    assert len(plans) == 1
    assert plans[0]["prompt"] == 1000 and plans[0]["base"] == 0
    assert plans[0]["spans"] == [{"src": 832, "dst": 640, "tok": 128},
                                {"src": 960, "dst": 832, "tok": 128}]
    assert plans[0]["ranges"] == [(0, 640), (768, 832), (960, 1000)]
    assert plans[0]["ranges_reported"] == 3
    assert scr_e2e_check.check_relocation_plan(plans[0], 2) == []


# ---------------------------------------------------------------------------------------------
# Flag plumbing
# ---------------------------------------------------------------------------------------------
def test_scr_flags_append_the_cap_after_the_mode() -> None:
    class Args:
        scr_mode = "segmented"
        scr_max_spans = 1

    assert scr_e2e_check.scr_flags(Args()) == ["--scr", "--scr-mode", "segmented",
                                               "--scr-max-spans", "1"]

    Args.scr_max_spans = None
    assert scr_e2e_check.scr_flags(Args()) == ["--scr", "--scr-mode", "segmented"]
