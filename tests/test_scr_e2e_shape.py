"""Host-side checks for the SCR e2e script's fixtures and plan-shape assertions.

The interleaved fixture exists to settle one question that no run has answered: whether segmented
relocation does anything tail relocation does not. Both selectors return the same span on every
prompt built before this shape, so both modes measured identically and the interleaved claim was
reasoned rather than observed. These tests prove the *shape* of the fixture on any machine -- two
deletions leaving two separated survivors -- using ports of the detector and the selectors that
`src/targets/qwen3_6/impl/runtime/prompt_edit.{h,cpp}` implements, driven by the script's own
alignment search.

The off-grid fixture is the other side of the same coin: an edit whose token delta is not a whole
number of pages, which the block detector cannot see at all and which forces the copied route. Its
shape is checked the same way, against a token-level detector port and the script's inverse
alignment search.

No server, no GPU, no tokenizer assets: the alignment search is checked against a deterministic
stand-in that counts words, and the planner arithmetic against token sequences built from it.
"""

from __future__ import annotations

import bisect
import contextlib

import bisect

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
    """A port of detail::plan_relocations' selection.

    The longest survivors are taken first, so a destination-page collision displaces the least reuse
    and the cap keeps the most. The cap counts relocated spans, not candidates: a survivor that
    collides with a longer one already kept is passed over rather than displacing it. At most one span
    may occupy a 64-token destination page, because the packed range plane names one relocated interval
    per page and the runtime refuses a plan that breaks the rule rather than truncating it.

    Both the collision rule and the reordering are inert on the aligned fixtures here, whose survivors
    begin on block boundaries and are separated by a whole block; they exist for the all-offset
    detector's off-grid destinations, which are the only source of a span that is not a whole page.
    The survivors are handed back in prompt order, which is the order the runtime consumes
    (`plan_relocations` sorts by `dest_begin` last).
    """
    pool = sorted(candidates(spans, resident_tokens, limit), key=lambda span: (-span[1], span[0]))
    kept = []
    for dest, tokens in pool:
        if len(kept) >= max_spans:
            break
        last = (dest + tokens - 1) // BLOCK
        if any(dest // BLOCK <= (kept_dest + kept_tokens - 1) // BLOCK and
               kept_dest // BLOCK <= last for kept_dest, kept_tokens in kept):
            continue
        kept.append((dest, tokens))
    return sorted(kept)


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
    # The planner's own line runs alongside the applied one on every request that reaches the SCR
    # block, and it names spans and counts that were planned rather than run. It must be skipped.
    planned = ("[scr-diag] plan mode=segmented shift=on prompt_tokens=1000 reuse_base=0 "
               "resident_kv=1024 max_spans=6 limit=984 boundary=no spans_seen=3 candidates=2 "
               "selector=2 fallback=no after_trim=2 expressible=yes final=2 relocated=256 "
               "reject{leading span is prefix reuse old=[0,64) new_begin=0 tokens=64}")
    plans = scr_e2e_check.parse_relocation_plans(planned + "\n" + line)
    assert len(plans) == 1
    assert plans[0]["prompt"] == 1000 and plans[0]["base"] == 0
    assert plans[0]["spans"] == [{"src": 832, "dst": 640, "tok": 128},
                                {"src": 960, "dst": 832, "tok": 128}]
    assert plans[0]["ranges"] == [(0, 640), (768, 832), (960, 1000)]
    assert plans[0]["ranges_reported"] == 3
    assert scr_e2e_check.check_relocation_plan(plans[0], 2) == []


def test_a_refusal_is_not_read_as_an_applied_plan() -> None:
    """A refusal names the spans and ranges the runtime *did not* run, so reading it back as a plan
    would assert a plan shape that never executed."""
    refused = ("[scr] refused an inexpressible relocation plan: refused=spans share a destination "
               "page reuse_base=0 prompt_tokens=1000 span{src=0 dst=64 tokens=100} "
               "span{src=256 dst=164 tokens=16} range[0,64) range[180,1000)")
    assert scr_e2e_check.parse_relocation_plans(refused) == []


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


# ---------------------------------------------------------------------------------------------
# What it would take to stop requiring the *source* to be page aligned
#
# Shipped relocation re-points a destination page at the source page that already holds its content,
# so both ends have to sit on the 64-token grid -- and `diff_prompt_edit` can only match whole
# blocks, so an edit whose token delta is not a multiple of 64 leaves the detector *nothing* to
# match after the edit. The pair below is a port of a proposed design: a token-level detector, and a
# selector that quantizes only the destination. Everything in this section is a port, held beside
# the shipped ports above so the difference between the two is one screen apart.
# ---------------------------------------------------------------------------------------------
ANCHOR_TOKENS = BLOCK          # detail::kPromptEditBlockTokens, the anchor production passes
PROBE_CAP = 8                  # detail::kMaxCandidateProbes
MIN_CHAIN_TOKENS = 2 * BLOCK   # detail::kMinimumChainTokens
MAX_CHAINS = 8                 # detail::kMaxDetectedSpans


def max_weight_chain(candidates):
    """One maximum-weight set of candidates monotone in both prompts (detail's single chaining pass).

    `candidates` are `(old_begin, new_begin, tokens)`. The chain is chosen by destination order while a
    source-order pointer admits only candidates whose source range ends at or before the current one
    begins, so both coordinates increase along it and no two of its spans can overlap on either axis.
    """
    if not candidates:
        return []
    order = sorted(candidates, key=lambda c: (c[1], c[0], -c[2]))
    weight = [0] * len(order)
    parent = [None] * len(order)
    source_keys = sorted({c[0] for c in order} | {c[0] + c[2] for c in order})

    def key_of(token):
        return bisect.bisect_left(source_keys, token) + 1

    tree_w = [0] * (len(source_keys) + 1)
    tree_from = [None] * (len(source_keys) + 1)
    best_w, best_at, cursor = 0, None, 0
    for pos, cand in enumerate(order):
        old_begin, _new_begin, tokens = cand
        while cursor < len(order) and order[cursor][0] + order[cursor][2] <= old_begin:
            node = key_of(order[cursor][0] + order[cursor][2])
            while node <= len(source_keys):
                if weight[cursor] > tree_w[node]:
                    tree_w[node], tree_from[node] = weight[cursor], cursor
                node += node & (~node + 1)
            cursor += 1
        inherited, inherited_from = 0, None
        if old_begin != 0:
            node = key_of(old_begin) - 1
            while node != 0:
                if tree_w[node] > inherited:
                    inherited, inherited_from = tree_w[node], tree_from[node]
                node -= node & (~node + 1)
        weight[pos] = inherited + tokens
        parent[pos] = inherited_from
        if weight[pos] > best_w:
            best_w, best_at = weight[pos], pos
    chain = []
    pos = best_at
    while pos is not None:
        chain.append(order[pos])
        pos = parent[pos]
    return list(reversed(chain))


def channel_spans(o, n):
    """The port of detail::diff_prompt_edit_shifted: claim-based seeding, then chained residual passes.

    Pass 1 finds the longest chain; everything it chose, and everything conflicting with it on either
    axis, leaves the pool; the next pass chains what is left. That is what recovers a crossing -- a run
    that sits above another in one prompt and below it in the other -- which no single monotone walk can
    represent. The union is disjoint in both prompts by construction, which is the invariant the
    selectors and `plan_page_remap` rely on.
    """
    anchor, probes = ANCHOR_TOKENS, PROBE_CAP
    if anchor <= 0 or len(o) < anchor or len(n) < anchor:
        return []

    index = {}
    for offset in range(len(o) - anchor + 1):
        index.setdefault(tuple(o[offset:offset + anchor]), []).append(offset)

    candidates = []
    claimed_to, at = 0, 0
    while at <= len(n) - anchor:
        if at < claimed_to:
            at = claimed_to
            continue
        found = index.get(tuple(n[at:at + anchor]))
        seeded = False
        if found is not None:
            for old_begin in found[:probes]:
                if o[old_begin:old_begin + anchor] != n[at:at + anchor]:
                    continue
                right = anchor
                while (old_begin + right < len(o) and at + right < len(n) and
                       o[old_begin + right] == n[at + right]):
                    right += 1
                candidates.append((old_begin, at, right))
                claimed_to = at + right
                seeded = True
                break
        if not seeded:
            at += 1

    chosen, pooled = [], list(candidates)
    for _pass in range(MAX_CHAINS):
        chain = max_weight_chain(pooled)
        if not chain or sum(c[2] for c in chain) < MIN_CHAIN_TOKENS:
            break
        chosen.extend(chain)
        pooled = [c for c in pooled if all(
            c[1] + c[2] <= k[1] or k[1] + k[2] <= c[1] or c[0] + c[2] <= k[0] or k[0] + k[2] <= c[0]
            for k in chain)]

    # Merge runs that touch on both axes. Per chain in the C++ (a fusion across chains would rebuild the
    # crossing they exist to express), which is the same thing here because a fused pair is always
    # adjacent in the sorted order.
    chosen.sort(key=lambda c: (c[1], c[0]))
    merged = []
    for span in chosen:
        if merged and merged[-1][1] + merged[-1][2] == span[1] and \
                merged[-1][0] + merged[-1][2] == span[0]:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + span[2])
        else:
            merged.append(span)
    return merged


def survivor_runs(previous_tokens, current_tokens):
    """The shipped all-offset detector, under the name the rest of this file already uses."""
    return channel_spans(previous_tokens, current_tokens)


def relocate_whole_pages(runs, resident_tokens, limit, page=BLOCK):
    """Quantize each run's *destination* to whole pages, and leave the source where it is.

    Only the destination has to be page granular: a page is what the allocation, the accounting and
    the per-page RoPE delta are expressed in, and a page holding both relocated and prefilled
    tokens could carry no single delta. The source is free to start mid-page, which is exactly the
    freedom a shifted edit needs; the price is that its payload has to be copied rather than
    re-pointed. Returns `(source_begin, dest_begin, tokens)` -- `dest_begin` and `tokens` are whole
    pages, `source_begin` deliberately is not.
    """
    out = []
    for old_begin, new_begin, tokens in runs:
        if old_begin == 0 and new_begin == 0:
            continue  # the leading run is ordinary prefix reuse
        if old_begin + tokens > resident_tokens or tokens == 0:
            continue
        dest_begin = -(-new_begin // page) * page  # the first whole destination page in the run
        room = min(new_begin + tokens, limit) - dest_begin
        if room < page:
            continue  # no whole page survives: nothing this run can relocate
        dest_end = dest_begin + (room // page) * page
        out.append((old_begin + (dest_begin - new_begin), dest_begin, dest_end - dest_begin))
    return sorted(out, key=lambda span: span[1])


def shift_scenario(delta, head=200, run_tokens=320, tail=64):
    """`delta` tokens inserted above an otherwise identical run: the shape every real edit has."""
    head_tokens = list(range(head))
    survivors = list(range(1000, 1000 + run_tokens))
    previous = head_tokens + survivors + list(range(2000, 2000 + tail))
    current = (head_tokens + [77770 + i for i in range(delta)] + survivors +
               list(range(3000, 3000 + tail)))
    return previous, current


def relocated_tokens(selected):
    return sum(span[-1] for span in selected)


# ---------------------------------------------------------------------------------------------
# The crossing fixture: the shape every turn of a context-editing client produces
#
# `shift_scenario` builds its survivors from a single run of distinct ascending integers, so the first
# anchor probe always succeeds, nothing competes for a source, and no run is ever disconnected from the
# prefix. Every possible detector agrees on it, which is why it could not distinguish a 16-token anchor
# from a 64-token one, or a greedy walk from a coverage one. This fixture is the opposite: the previous
# prompt carries an *echo* of the block the client rewrote, placed after the transcript, so the rewritten
# block has a real copy whose source sits above the transcript's while its destination sits below it.
# ---------------------------------------------------------------------------------------------
def crossing_scenario(transcript=2048, ledger=256, prefix=640, tail=128):
    """`previous` = prefix, old ledger, transcript, echo of the new ledger, tail.

    `current` = prefix, new ledger, the same transcript. That is request 131 of the field trace with the
    token counts scaled down; the ordering, which is what the failure is about, is unchanged.
    """
    def block(label, count):
        return [label * 1000000 + i for i in range(count)]

    shared_prefix = block(1, prefix)
    ledger_old = block(2, ledger)
    ledger_new = block(3, ledger)
    shared = block(4, transcript)
    previous = shared_prefix + ledger_old + shared + ledger_new + block(5, tail)
    current = shared_prefix + ledger_new + shared
    return previous, current


def test_the_crossing_fixture_is_the_shape_the_bug_needs() -> None:
    """Guard the fixture itself: a prefix, a moved block whose source sits above the transcript, and a
    transcript below it in the destination."""
    previous, current = crossing_scenario()
    transcript_at = 640 + 256

    assert previous[:640] == current[:640], "the prompts share a prefix"
    assert previous[640:896] != current[640:896], "the ledger differs at the same position"
    echo_at = 640 + 256 + 2048
    assert previous[echo_at:echo_at + 256] == current[640:896], \
        "the new ledger is echoed in the previous prompt, which is what makes the match cross"
    assert 640 + 256 < echo_at, "the echo sits above the transcript, not below it"
    assert transcript_at == 896, "the shared transcript starts where the old ledger ended"


def test_a_coverage_detector_keeps_the_transcript_a_greedy_walk_loses() -> None:
    """The measurement the rewrite is for, over the shape above.

    A monotone walk commits to the echoed ledger on its first match and then cannot reach the transcript
    behind it: 896 of 2944 tokens. Coverage chaining sees both, because the transcript survives as a
    second chain rather than being unreachable.
    """
    previous, current = crossing_scenario()

    greedy = [(0, 0, 640), (640 + 256 + 2048, 640, 256)]
    assert sum(t for _o, _n, t in greedy) == 896, "the walk's own answer, for the record"

    spans = survivor_runs(previous, current)
    covered = sum(t for _o, _n, t in spans)
    assert covered == 640 + 256 + 2048, f"coverage keeps the transcript too: {spans}"
    assert (896, 896, 2048) in spans, f"the transcript is reported at its true offsets: {spans}"
    assert 2048 > 256 * 3, "the run the walk loses is the largest one available"


def test_every_reported_span_is_disjoint_in_both_prompts() -> None:
    """Now that monotonicity is gone, disjointness in both prompts is the whole contract.

    `plan_page_remap` refuses a plan that names one physical page twice and the runtime reports that
    refusal as a failed request, so a detector that emitted two spans sharing a source page would turn a
    degraded plan into an error.
    """
    for previous, current in (crossing_scenario(),
                              crossing_scenario(transcript=320, ledger=192, prefix=192, tail=64)):
        spans = survivor_runs(previous, current)
        for i, (o, n, t) in enumerate(spans):
            assert t > 0
            for o2, n2, t2 in spans[i + 1:]:
                assert n + t <= n2 or n2 + t2 <= n, f"destination overlap: {spans}"
                assert o + t <= o2 or o2 + t2 <= o, f"source overlap: {spans}"
        assert sum(t for _o, _n, t in spans) > 0


def test_a_reordering_is_covered_rather_than_half_prefilled() -> None:
    """A swap is the case no single monotone chain can hold: the source order and the destination order
    disagree. It has to cost a second chain, not half the prompt."""
    head = [1000000 + i for i in range(64)]
    a = [2000000 + i for i in range(192)]
    b = [3000000 + i for i in range(192)]
    tail = [4000000 + i for i in range(64)]
    previous = head + a + b + tail
    current = head + b + a + tail

    spans = survivor_runs(previous, current)
    assert sum(t for _o, _n, t in spans) == len(previous), \
        f"every token survives a swap: {spans}"
    assert len(spans) == 4, f"reported as the two moved halves, not one run: {spans}"


def test_a_three_token_edit_leaves_the_shipped_detector_nothing_to_match() -> None:
    """The cliff the alignment requirement puts under every realistically typed edit."""
    previous, current = shift_scenario(5)
    shipped = select_segmented(surviving_spans(previous, current), len(previous),
                               len(current) - 1, 6)
    assert shipped == [], \
        "nothing on the 64-grid survives a 5-token edit, so there is nothing to re-point"

    proposed = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                    len(current) - 1)
    assert len(proposed) == 1, f"the token-level detector finds the run: {proposed}"
    source, dest, tokens = proposed[0]
    assert source % BLOCK != 0, "the source is deliberately off-grid; that is the whole point"
    assert dest % BLOCK == 0 and tokens % BLOCK == 0 and tokens >= BLOCK
    assert tokens >= 320 - (BLOCK - 1) - (BLOCK - 1), \
        f"only the two edge pages are lost, got {tokens}"


def test_the_aligned_case_still_re_points_the_page_and_copies_nothing() -> None:
    """A delta that is a multiple of the page size must stay on the existing fast path: the
    destination already names a page that holds the content, so the new scheme is a superset."""
    previous, current = shift_scenario(128, head=192)
    shipped = select_segmented(surviving_spans(previous, current), len(previous),
                               len(current) - 1, 6)
    proposed = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                    len(current) - 1)
    assert [(dest, tokens) for _source, dest, tokens in proposed] == shipped
    assert proposed[0][0] % BLOCK == 0, "an aligned edit needs no copy: source is on the grid too"
    assert relocated_tokens(shipped) == 320


def test_every_shift_relocates_all_but_the_two_edge_pages() -> None:
    """The measurement the change is for: coverage over the whole shift space, shipped vs proposed."""
    shipped_total = 0
    proposed_totals = []
    for delta in range(1, BLOCK):
        previous, current = shift_scenario(delta)
        shipped = select_segmented(surviving_spans(previous, current), len(previous),
                                   len(current) - 1, 6)
        proposed = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                        len(current) - 1)
        shipped_total += relocated_tokens(shipped)
        proposed_totals.append(relocated_tokens(proposed))
        # One span either way, and the token-level detector is the only one that finds it.
        assert len(proposed) == 1, f"delta={delta}: {proposed}"
        assert min(proposed_totals) >= 320 - 126, f"delta={delta} lost more than the edge pages"
        assert max(proposed_totals) <= 320
    assert shipped_total == 0, "the shipped detector relocates nothing for any of these edits"
    mean = sum(proposed_totals) / len(proposed_totals)
    print(f"shift 1..{BLOCK - 1}: shipped relocates 0 tokens; proposed mean {mean:.1f} of 320")
    assert mean >= 250, f"a mean of {mean:.1f} of 320 is not the edge-page-only loss"


def test_the_proposed_spans_still_tile_the_suffix() -> None:
    """The planner's contract has to survive the change: prefilled or relocated, never both."""
    previous, current = shift_scenario(5)
    proposed = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                    len(current) - 1)
    ranges = prefilled_ranges([(dest, tokens) for _s, dest, tokens in proposed], len(current))
    tiled = sum(end - begin for begin, end in ranges) + relocated_tokens(proposed)
    assert tiled == len(current), f"{tiled} != {len(current)}: {ranges} + {proposed}"
    assert ranges[0][0] == 0, "an in-the-middle edit reuses no prefix, so the head is prefilled"


def test_destination_quantization_costs_only_the_two_edge_pages() -> None:
    """The residual cost of keeping the *destination* on the page grid, over every shift.

    A copy can write anywhere, so the destination is the only end that has to stay aligned -- and it
    is not a copy constraint: a page carries exactly one RoPE delta, so a page holding both relocated
    and prefilled tokens could carry no single shift. The price is a page at each end of a run, which
    is the same fringe the shipped path already trims on an *aligned* edit: the source-side
    relaxation neither adds nor removes it. Measuring it says where the remaining loss sits: for the
    runs that matter it is a few percent, and only a run shorter than about two pages can still
    relocate nothing.
    """
    def mean_fraction(run_tokens):
        totals = []
        for delta in range(BLOCK):
            previous, current = shift_scenario(delta, run_tokens=run_tokens)
            spans = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                         len(current) - 1)
            totals.append(relocated_tokens(spans))
            assert max(totals) <= run_tokens, "a span cannot relocate more than the run holds"
        return sum(totals) / len(totals) / run_tokens

    one_page = mean_fraction(64)
    two_pages = mean_fraction(128)
    long_run = mean_fraction(4096)
    print(f"mean relocated fraction over every shift: 64->{one_page:.2f} 128->{two_pages:.2f} "
          f"4096->{long_run:.2f}")
    assert one_page < 0.1, "a one-page run has almost nowhere to land"
    assert two_pages >= 0.45, "a two-page run keeps about one page"
    assert long_run >= 0.95, "a long run loses only its two edge pages"


def test_a_page_of_a_relocated_span_needs_at_most_two_copies_and_one_delta() -> None:
    """Why the source is free: a page's content comes from one contiguous source range, so it is at
    most two device copies, and every token it holds shares one shift -- which is what lets the
    per-page RoPE delta plane stay exactly as it is."""
    previous, current = shift_scenario(5)
    source, dest, tokens = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                                len(current) - 1)[0]
    shift = dest - source
    pages = {}
    for offset in range(0, tokens, BLOCK):
        src, remaining = source + offset, BLOCK
        segments = []
        while remaining:
            src_page, src_offset = divmod(src, BLOCK)
            take = min(remaining, BLOCK - src_offset)
            dst = dest + offset + (BLOCK - remaining)
            # Every segment shifts by the span's own shift: one delta covers the whole page.
            assert (dst // BLOCK) * BLOCK + dst % BLOCK - (src_page * BLOCK + src_offset) == shift
            segments.append((src_page, src_offset, (dst - shift) % BLOCK, take))
            src += take
            remaining -= take
        pages[dest + offset] = segments

    assert set(pages) == set(range(dest, dest + tokens, BLOCK)), "one entry per destination page"
    for page, segments in pages.items():
        assert sum(take for _s, _o, _d, take in segments) == BLOCK, f"page {page} is not filled"
        assert len(segments) <= 2, f"page {page} needs {len(segments)} copies"
        for _src_page, src_offset, _dst_offset, take in segments:
            assert 0 < take <= BLOCK - src_offset, "a segment stays inside its source page"


# ---------------------------------------------------------------------------------------------
# The off-grid fixture: an edit the page grid cannot express
#
# Scope A's edit, with the middle sized so the delta is *not* a multiple of 64. The block detector
# then matches nothing after the edit, and the surviving source misses the destination's page offset
# by construction, so the re-pointing route cannot express it and the copied route is the only one
# that can. That is the shape `--shape off-grid` runs and `source_may_shift` asserts.
# ---------------------------------------------------------------------------------------------
def test_the_off_grid_fixture_moves_the_edit_off_the_page_grid() -> None:
    with live_shape("off-grid"):
        pads = scr_e2e_check.solve_alignment(None, 512, 900, False)
        assert pads is not None, "the off-grid search failed"
        vocab = {}
        text_a = scr_e2e_check.request_prompt("a", pads)
        text_b = scr_e2e_check.request_prompt("b", pads)
    prompt_a = server_tokens(text_a, vocab)
    prompt_b = server_tokens(text_b, vocab)

    delta = len(prompt_a) - len(prompt_b)
    assert delta > 0, "the fixture is a deletion"
    assert delta % BLOCK != 0, \
        f"the edit moves {delta} tokens, a whole number of pages: that is not the off-grid case"

    # A whole-block index only matches content on the same grid in both prompts, so after an
    # off-grid edit it finds nothing: that is why the token-level detector exists at all.
    aligned = surviving_spans(prompt_a, prompt_b)
    assert all((old - new) % BLOCK == 0 for old, new, _t in aligned), \
        f"the block detector found an off-grid survivor, so the fixture is not off-grid: {aligned}"

    runs = survivor_runs(prompt_a, prompt_b)
    assert len(runs) >= 2, f"expected the shared prefix and the survivor, got {runs}"
    old_begin, new_begin, tokens = runs[-1]
    assert old_begin - new_begin == delta, \
        "the surviving run is shifted by exactly the edit's token delta"
    assert (old_begin - new_begin) % BLOCK != 0, "the source has to start off the page grid"
    assert tokens >= 2 * BLOCK, "the survivor has to be long enough to relocate whole pages"


def test_the_off_grid_plan_check_wants_a_source_off_the_page_grid() -> None:
    """A plan whose source shares its destination's page offset is the re-pointing shape. The
    off-grid check must reject it, or a run that silently fell back would report a pass."""
    def plan(src):
        return {
            "prompt": 1000,
            "base": 0,
            "spans": [{"src": src, "dst": 640, "tok": 128}],
            "ranges": [(0, 640), (768, 1000)],
            "ranges_reported": 2,
        }

    on_grid = scr_e2e_check.check_relocation_plan(plan(832), 1, source_may_shift=True)
    assert any("off-grid" in problem for problem in on_grid), on_grid

    off_grid = scr_e2e_check.check_relocation_plan(plan(833), 1, source_may_shift=True)
    assert off_grid == [], off_grid

    # Without the relaxation the same span is the whole-block rule's business, which is what every
    # aligned fixture relies on.
    strict = scr_e2e_check.check_relocation_plan(plan(833), 1)
    assert any("64-token blocks" in problem for problem in strict), strict


def test_an_off_grid_plan_from_the_port_passes_the_shape_check() -> None:
    """Tie the check to the port it is written for: the plan the token-level selector produces on a
    shifted edit is exactly the plan `check_relocation_plan(..., source_may_shift=True)` accepts."""
    previous, current = shift_scenario(5)
    selected = relocate_whole_pages(survivor_runs(previous, current), len(previous),
                                    len(current) - 1)
    assert len(selected) == 1, f"expected one off-grid span, got {selected}"
    source, dest, tokens = selected[0]
    ranges = prefilled_ranges([(dest, tokens)], len(current))
    plan = {
        "prompt": len(current),
        "base": 0,
        "spans": [{"src": source, "dst": dest, "tok": tokens}],
        "ranges": ranges,
        "ranges_reported": len(ranges),
    }
    assert scr_e2e_check.check_relocation_plan(plan, 1, source_may_shift=True) == [], plan
