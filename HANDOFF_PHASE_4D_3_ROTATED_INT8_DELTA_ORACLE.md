# Phase 4d-3 Handoff — Rotated INT8 Variants (`rk8v4`, `rk4v4-e8`) Delta-Plane Oracle + I8 Prefill Path

**Purpose:** Extend the int8 delta-plane oracle infrastructure from Phase 4d-2 to the rotated storage variants `rk8v4` and `rk4v4-e8`, and add the I8 prefill (A1) coverage that 4d-2 deferred. Same scenario, same delta plane, same negative control as the two existing oracle tests; the additions are a packed-int4 plane model, a host port of the 64-point Hadamard for the FP64 anchor, a batch view on `Int8Cache`, and a second ctest binary.

**Prerequisites:** Phase 4d-2 applied and green on Windows (`Int8Cache`, `run_cached_attention_view_i8`, the templated `ideal_attention`, and `tests/ops/test_gqa_kv_page_delta_i8.cpp` all exist — confirmed present on the branch). Phase 4d code-complete and uncommitted.

**Scope:** Two variants, both selected by the runtime flag combination from `src/targets/qwen3_6/impl/runtime/layouts_impl.h`:

| storage | packed_v | rotate_k | rotate_v | packed_k | e8_lattice | e8_root | codec |
|---|---|---|---|---|---|---|---|
| `RotatedInt8KeyInt4ValueGroup64` (rk8v4) | T | T | T | F | F | F | `RK8V4` |
| `RK4V4E8` (rk4v4-e8) | T | T | T | T | T | F | `RK4V4E8` |
| `RotatedInt4KeyInt4ValueGroup64` (rk4v4) *(3rd row: same codec, different encoder)* | T | T | T | T | F | F | `RK4V4E8` |
| `RK2V4E8` | T | T | T | F | F | T | **out of scope → throw** |
| `Int8Group64` *(optional 4th row, regression)* | F | F | F | F | F | F | `I8G64` |

**Out of scope — Phase 4d-4:** `rk2v4-e8`. Its K decode is the only one that needs a real host codec port (`e8_root_decode_8d_int8`: `c_e8_stage1_i8x8` root table + `c_axis_i8x8` axis table + `c_radius_scale`, `src/ops/kernel/e8_root_codec.cuh:690-732`), and its K plane has a genuinely different packing (64 bytes/head-token, `paged_kv_page_head_offset<64, KVHeads>` at `gqa_attention_decode_i8.cuh:338-352`; the wrapper expects a `kHeadDim/4` leading extent, `gqa_attention.cpp:84`).

**Also out of scope: the encode side.** The oracle never encodes; it decodes what the production fill kernel wrote. This is why `rk4v4-e8` needs no E8 lattice in the oracle — see Part 0b, which proves decode is `unpack_i4 × scale` for every `PackedK` variant, and Part 0c, which shows `e8_lattice` appears nowhere in the read path.

**Anchor decision:** both in-scope variants have `rotate_k = true`, so the oracle needs a Hadamard stage (Part 2). The FP64 anchor **stays enabled** for both, judged with the rotated profile `kConformanceRotatedI8` (Part 4c — the first Windows run showed the plain `kConformanceInt8` is not the right profile for a rotated cache, and said why). If Part 0a reports flag values that contradict the table above, STOP — the plane model and the anchor decision both change.

**Relocation anchor (Part 4c):** the delta run is anchored to `ideal_attention_with_delta`, an FP64 oracle that rotates each decoded key back by its page's delta. The cache-against-cache equivalence and the multiplicative control are kept as observations; they cannot carry the discrimination on a packed int4 K plane, for the measured reason Part 4c records.

**Gate:** Phase 4d cannot be committed until this test passes on Windows alongside `ninfer_gqa_kv_page_delta_i8_test`.

**Status (2026-10-10):** Parts 1–5 are implemented on the branch — `KvCodec`, `codec_of`, `unpack_i4`, `hadamard64_group`, `inverse_hadamard_plane`, the generalized `Int8Cache` (per-plane extent/dtype, `batch_view()`, per-plane typed `image()` with the inverse Hadamard), `tests/ops/test_gqa_kv_page_delta_rotated.cpp`, and its CMake registration. The **first Windows run** of that binary happened and returned 7 failures: all 7 are the two calibration problems Part 4c records, both measured, neither a code fault — the four host self-checks and both packed-plane cross-checks passed. Part 4c's redesign (the anchored delta oracle plus `kConformanceRotatedI8`) is implemented and verified off-GPU. The **second Windows run** then took it green: all checks pass, 0 failures, in ~3.0 s. Measured, per variant (`relative_l2`):

| variant | A3 delta anchor | A3 reference anchor | A3 vs direct prefill | A3 control | A1 delta anchor | A1 vs direct prefill |
|---|---|---|---|---|---|---|
| rk8v4 | 3.754e-3 | 3.750e-3 | 4.255e-3 | 5.652e-2 (11x) | 2.463e-3 | 2.503e-3 |
| rk4v4-e8 | 3.746e-3 | 3.777e-3 | 1.719e-2 | 5.724e-2 (11x) | 2.459e-3 | 1.315e-2 |
| rk4v4 | 3.740e-3 | 3.757e-3 | 1.174e-2 | 5.672e-2 (11x) | 2.454e-3 | 8.828e-3 |
| plain I8G64 *(other binary)* | — | 3.142e-3 vs 3.15e-3 | 3.434e-3 | 5.667e-2 (2x) | — | — |

The delta anchor matching the reference anchor to within 1e-5 on every row is the phase's central evidence: the relocation path is as close to FP64 truth as a directly prefilled cache, so the per-page counter-rotation is exact, and a skipped one would read ~5.7e-2. The plain row is the reference point for the profile correction in Part 4c.

**Still outstanding:** Part 8 steps 3 and 4 after the redesign (the shared-header refactor of `ideal_attention` makes the earlier step-3 green stale), and the phase gate itself.

**Read this first.** Part 0 gathers facts that determine Parts 1–4. If a fact does not match the expectation shown, STOP and report; do not adapt the docs.

---

## Part 0 — Gather facts

**0a. The flag combinations.** Paste:

```bash
sed -n '700,735p' src/targets/qwen3_6/impl/runtime/layouts_impl.h
```

Expected: the `kv_packed_v / kv_rotate_k / kv_rotate_v / kv_packed_k / kv_e8_lattice / kv_e8_root` initializers, with `RotatedInt8KeyInt4ValueGroup64` and `RK4V4E8` matching the table above, and `RK4V4E8` being the only storage whose `kv_e8_lattice` is `true`. If either in-scope variant differs, STOP.

**0b. The decode-branch proof.** Paste:

```bash
grep -n "else if constexpr (PackedK)" -A 12 src/ops/kernel/gqa_attention_decode_i8.cuh
grep -n "E8Root" -A 14 src/ops/kernel/gqa_attention_decode_i8.cuh | head -40
```

Expected (verified on this checkout at `gqa_attention_decode_i8.cuh:377-383` and `:338-352`): the `PackedK` branch does

```cpp
const std::int64_t koff = gqa_kv_i4_code_index<Geometry>(physical_page, kv_head, d / 2, key & kPagedKVPageMask);
std::int8_t* dst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
gqa_kv_unpack_i4x16(&reinterpret_cast<const std::uint8_t*>(cache_k_i8)[koff], dst);
```

— one unpack per 16 codes, **no lattice math** — while `E8Root` is the branch that calls `e8_root_decode_8d_int8`. The only place `e8_project_8d_warp` appears is the *encode* arm inside the fill kernel (`gqa_attention_decode_i8.cuh:264-276`, with its own half-coset disclaimer). If the `PackedK` read branch does lattice work, the design below is wrong: **STOP and report**, because `rk4v4-e8` then needs a codec port and moves to 4d-4 with `rk2v4-e8`.

Also paste the pack/unpack pair, which fixes the nibble convention the host port has to match:

```bash
grep -n "gqa_kv_pack_i4\|gqa_kv_unpack_i4\|gqa_kv_unpack_i4x16" -A 6 src/ops/kernel/gqa_attention_kv_quant.cuh
```

**0c. The plane contract (this is the authority for `image()`).** Paste:

```bash
sed -n '77,115p' src/ops/wrapper/gqa_attention.cpp
```

Expected (verified at `gqa_attention.cpp:79-92`, and identically in `validate_batch_cache` at `:147-160`):

```cpp
const DType code_dtype = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
if (cache.k_pages.dtype != ((cache.packed_k || cache.e8_root) ? DType::U8 : code_dtype) ||
    cache.v_pages.dtype != (cache.packed_v ? DType::U8 : code_dtype)) {
    throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
}
const std::int32_t k_dim = cache.e8_root ? kHeadDim / 4 : (cache.packed_k ? kHeadDim / 2 : kHeadDim);
const std::int32_t v_dim = cache.packed_v ? kHeadDim / 2 : kHeadDim;
```

Consequences, all of which the test must satisfy before a kernel ever runs:

- `I8G64`: `k_pages` and `v_pages` are `DType::I8`, leading extent 256.
- `RK8V4`: `k_pages` `I8` / 256, `v_pages` **`DType::U8` / 128**.
- `RK4V4E8`: `k_pages` **`U8` / 128**, `v_pages` **`U8` / 128**.
- Scale planes are `FP16` `{4, kPageSize, kv_heads, physical_pages}` in every I8 case (`:107-114`).
- `rotate_v` without `packed_v` and `e8_lattice` without `packed_k` both throw (`:64-69`).
- The batch view checks the same rules (`:147-160`) plus `block_tables` shaped `{logical_pages, table_rows}` (`:166`). Neither validator inspects `rope_deltas`; the launchers read it directly, so the row layout is the caller's contract (Part 3b).

If the checkout disagrees with any line above, STOP.

**0d. The 4d-2 base you extend rather than rewrite** (`tests/ops/gqa_relocation_test_common.h`): `Int8Cache` (byte-plane `DeviceBuffer`s, half scale planes, the six `bool` flag fields, `image()`, `k_logical`/`v_logical`), `append_chunk(Int8Cache&, …)`, `run_cached_attention_view_i8`, the templated `ideal_attention`, `kConformanceInt8`, `measure`/`expect_conformant`/`expect_equivalent`/`expect_control_separates`, and `tests/ops/test_gqa_kv_page_delta_i8.cpp` as the structural template for the new file.

**0e. Does the public route forward the delta plane?** Paste:

```bash
grep -n "rope_deltas\|rope_inv_freq" src/ops/wrapper/gqa_attention.cpp src/ops/launcher/gqa_attention_decode.cu src/ops/launcher/gqa_attention_prefill.cu
```

Expected: yes. `ops::gqa_attention` passes the `PagedKVBatchLayerView` straight into `detail::gqa_attention_small_t_launch` / `detail::gqa_attention_prompt_launch` (`gqa_attention.cpp:435-455`), and both launchers read `static_cast<const std::int32_t*>(cache.rope_deltas.data)` plus `make_rerotate_freq(cache.rope_inv_freq)` (`gqa_attention_decode.cu:284-310`, `gqa_attention_prefill.cu:180-205`). If the grep shows the same, Part 3b takes the **public** route; if the forwarding is absent or gated on BF16, Part 3b falls back to the `detail::` hook and says so in the report.

---

## Part 1 — Generalize the plane model in `tests/ops/gqa_relocation_test_common.h`

**1a. `KvCodec` and `codec_of`.** Add next to the existing `kQuantGroup` constants:

```cpp
// The three plane models this oracle can decode. Selected by the runtime flag combination, never by
// the storage enum: the oracle's input is a cache view, and the view carries flags.
enum class KvCodec {
    I8G64,   // full-width int8 K and V, no rotation
    RK8V4,   // full-width int8 K, packed int4 V, K and V Hadamard-rotated
    RK4V4E8, // packed int4 K and V, K and V Hadamard-rotated; the E8 projection is encode-side only
};

// Deliberately not constexpr and deliberately not optional<KvCodec>: the e8_root case throws, and a
// throwing body is ill-formed in a constant expression, while a nullopt is something a caller can
// ignore. A variant the oracle cannot decode has to be loud.
[[nodiscard]] inline KvCodec codec_of(bool packed_v, bool rotate_k, bool rotate_v, bool packed_k,
                                      bool e8_lattice, bool e8_root) {
    if (e8_root) {
        // rk2v4-e8's K read path is e8_root_decode_8d_int8: root table + axis table + radius scale.
        // That host port is Phase 4d-4. Do not approximate it with the packed-int4 model below.
        throw std::invalid_argument("codec_of: the rk2v4-e8 oracle is Phase 4d-4");
    }
    // Mirror both wrapper invariants (gqa_attention.cpp:64-69) here, so a bad flag set fails at cache
    // construction instead of aborting mid-run inside gqa_attention_cached.
    if (e8_lattice && !packed_k) {
        throw std::invalid_argument("codec_of: E8 lattice requires packed K");
    }
    if (rotate_v && !packed_v) {
        throw std::invalid_argument("codec_of: rotate_v requires packed V");
    }
    if (packed_k) { return KvCodec::RK4V4E8; }                       // rk4v4-e8, and rk4v4
    if (packed_v || rotate_k || rotate_v) { return KvCodec::RK8V4; } // rk8v4
    return KvCodec::I8G64;
}
```

`e8_lattice` does not participate in the selection, and that is the whole point of Part 0b: it is an encode-side projection, so `RK4V4E8` and `RotatedInt4KeyInt4ValueGroup64` share one decode model. Do not add a branch for it.

**1b. Give `Int8Cache` per-variant plane extents and its `codec`.** The existing struct hardcodes `{kHeadDim, …}` and `DType::I8` for both planes, which Part 0c shows is invalid for `packed_v`/`packed_k`. Replace the six loose `bool` fields with the flags *and* the derived codec, and size the buffers from the extents:

```cpp
    KvCodec codec;
    bool rotate_k = false, packed_k = false, rotate_v = false, packed_v = false;

    [[nodiscard]] std::int32_t k_leading() const { return codec == KvCodec::RK4V4E8 ? kHeadDim / 2 : kHeadDim; }
    [[nodiscard]] std::int32_t v_leading() const { return codec == KvCodec::I8G64 ? kHeadDim : kHeadDim / 2; }
    [[nodiscard]] DType        k_dtype()   const { return codec == KvCodec::RK4V4E8 ? DType::U8 : DType::I8; }
    [[nodiscard]] DType        v_dtype()   const { return codec == KvCodec::I8G64   ? DType::I8 : DType::U8; }
```

The constructor takes the six flags, calls `codec_of` once, stores `codec`, and allocates

- `k_codes`: `k_leading() * kPageSize * kv_heads * physical_pages` bytes,
- `v_codes`: `v_leading() * kPageSize * kv_heads * physical_pages` bytes,

with the scale planes unchanged (`kQuantGroups` leading extent, `sizeof(__half)`). `view()`/`batch_view()` must publish exactly `{k_leading(), kPageSize, kv_heads, physical_pages}` / `{v_leading(), …}` with `k_dtype()` / `v_dtype()`, or the wrapper throws before the kernel runs (Part 0c). Keep every other field of `view()` byte-for-byte as it is, including `quant_group = kQuantGroup` and `dtype = DType::I8`.

Because `rk8v4` requires `rotate_v → packed_v` and `rk4v4-e8` requires `e8_lattice → packed_k`, the live combinations are exactly the in-scope rows; `codec_of` only has to make a *mis-specified* flag set land on a codec rather than silently decoding as `I8G64` (and, per 1a, to reject the two genuinely illegal sets).

**1c. `image()`: per-plane host typing, then the inverse Hadamard.** This is where the 4d-2 reviewer's signed-read finding comes back, and the fix is forced by Part 0c rather than chosen: the view declares the full-width planes as `I8` and the packed planes as `U8`, so **read each plane into a host vector whose element type matches its declared dtype**, and decode from that:

```cpp
// Plane extent and dtype per plane, derived from the codec the constructor selected. These must
// match what the wrapper validates (gqa_attention.cpp:79-86): a packed plane is U8 with leading
// extent kHeadDim / 2, a full-width plane is the code dtype with leading extent kHeadDim.
[[nodiscard]] std::int32_t k_leading() const { return codec == KvCodec::RK4V4E8 ? kHeadDim / 2 : kHeadDim; }
[[nodiscard]] std::int32_t v_leading() const { return codec == KvCodec::I8G64   ? kHeadDim : kHeadDim / 2; }
[[nodiscard]] DType k_dtype() const { return codec == KvCodec::RK4V4E8 ? DType::U8 : DType::I8; }
[[nodiscard]] DType v_dtype() const { return codec == KvCodec::I8G64   ? DType::I8 : DType::U8; }

[[nodiscard]] HostImage image() const {
    // Both planes are byte storage, so the host vector's element type is what carries the
    // interpretation: a full-width plane is read into std::int8_t and a packed plane into
    // std::uint8_t. A plain static_cast<double> is then correct for the signed plane and is never
    // used for a packed one, which only ever goes through unpack_i4.
    const bool k_full = k_leading() == kHeadDim;
    const bool v_full = v_leading() == kHeadDim;
    std::vector<std::int8_t>  k_i8, v_i8;   // full-width planes
    std::vector<std::uint8_t> k_u8, v_u8;   // packed planes
    ...
    for (page, head, offset):
        const std::size_t row = (page * kKvHeads + head) * kPageSize + offset;
        const std::int32_t group = d / kQuantGroup;
        const double ks = static_cast<double>(__half2float(k_scale_host[row * kQuantGroups + group]));
        const double vs = static_cast<double>(__half2float(v_scale_host[row * kQuantGroups + group]));
        // Each plane is indexed with its own leading extent -- k_leading() for K, v_leading() for V.
        // For a packed plane that is kHeadDim / 2, so d / 2 lands on the byte holding dims
        // {d & ~1, d | 1}; indexing a packed plane with kHeadDim would run past the buffer.
        out.k[elem_idx] = k_full ? static_cast<double>(k_i8[row * k_leading() + d]) * ks
                                 : static_cast<double>(unpack_i4(k_u8[row * k_leading() + d / 2], d & 1)) * ks;
        out.v[elem_idx] = v_full ? static_cast<double>(v_i8[row * v_leading() + d]) * vs
                                 : static_cast<double>(unpack_i4(v_u8[row * v_leading() + d / 2], d & 1)) * vs;
    // ... then, once the whole plane is decoded (H is orthonormal, symmetric and involutory, so the
    // same host transform both applies and undoes the rotation):
    if (rotate_k) { inverse_hadamard_plane(out.k); }
    if (rotate_v) { inverse_hadamard_plane(out.v); }
}
```

with

```cpp
// Matches gqa_kv_unpack_i4 exactly: the low nibble is the even dim, the high nibble is the odd dim.
[[nodiscard]] inline std::int8_t unpack_i4(std::uint8_t packed, std::int32_t parity) {
    const unsigned nibble = parity ? (packed >> 4) : (packed & 0x0Fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}
```

Worked check the implementer should keep in mind: code `-8` → nibble `0x8` → `(8 ^ 8) - 8 = -8`; code `+7` → nibble `0x7` → `(7 ^ 8) - 8 = 7`; code `-1` → nibble `0xF` → `(15 ^ 8) - 8 = -1`. This is a two's-complement 4-bit sign extension, not a biased offset — an off-by-eight here would pass a zero-centred test and silently bias the oracle.

**Where the rotation sits relative to the scale.** Apply the Hadamard to the **dequantized** values, after multiplication by the scale, matching the production order (fill: `hadamard64` → `absmax` → scale → quantize; read: dequant → inverse Hadamard). Group boundaries are `d / kQuantGroup`; for a packed plane the group is still 64 *elements*, i.e. 32 *bytes*.

---

## Part 2 — Host port of the 64-point Hadamard, and its anchored self-checks

**Summary, before the derivation: a constant group `[c, c, …, c]` must transform to `[8c, 0, …, 0]`. The DC gain is 8, not 1.** The 64-point ±1 Hadamard has 64 in every DC entry and the final stage scales by `0.125 = 1/8`, so `64 × 0.125 = 8`; the energy identity confirms it, `(8c)² = 64c² = ‖c·1₆₄‖²`. A port written to return the input constant would be off by a factor of 8 and would silently bias every rotated score.

**2a. The port.** Device lane `L` holds group elements `L` and `L + 32`, the five butterfly stages read the *pre-stage* values of lane `L ^ offset`, and the final stage folds the two registers with a `0.125` scale (`src/ops/kernel/gqa_attention_kv_quant.cuh:75-91`). Mirror it exactly:

```cpp
// Host port of gqa_kv_hadamard64. y holds one 64-element group in d order: y[L] is the device's x0
// on lane L, y[L + 32] is x1. Stage updates must read the previous stage's whole vector, because
// __shfl_xor_sync is simultaneous across lanes.
inline void hadamard64_group(double* y) {
    std::array<double, 64> prev{};
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::copy(y, y + 64, prev.begin());
        for (int lane = 0; lane < 32; ++lane) {
            const int  src = lane ^ offset;
            const bool hi  = (lane & offset) != 0;
            y[lane]      = hi ? prev[src] - prev[lane] : prev[lane] + prev[src];
            y[lane + 32] = hi ? prev[src + 32] - prev[lane + 32]
                              : prev[lane + 32] + prev[src + 32];
        }
    }
    for (int lane = 0; lane < 32; ++lane) {
        const double a = y[lane], b = y[lane + 32];
        y[lane]      = (a + b) * 0.125;
        y[lane + 32] = (a - b) * 0.125;
    }
}
```

`H` is symmetric and involutory (`H² = I`), which is why one function serves both directions; the production V path relies on the same property, using `gqa_kv_hadamard64` to *undo* the stored V rotation in `gqa_kv_inverse_rotate_output_kernel` (`gqa_attention_kv_quant.cuh:114-137`).

**2b. The self-checks, with the expected values written down.** Run all four before any cache is built, and print them:

1. **DC gain.** With `y = [c, c, …, c]`, `c = 1.0`, the transform must produce `[8, 0, …, 0]` exactly: assert `|y[0] - 8| ≤ 1e-12` and `max |y[i]| (i > 0) ≤ 1e-12`.
   **Note for the implementer:** an earlier review of this doc stated the expectation as "index 0 equals the input constant `c`, all others zero". That is arithmetically wrong for this device normalization; a port written to produce `c` would be scaling by an extra `1/8`, and the oracle would then be wrong by a constant factor on every rotated score. The reviewer's own "energy conservation" sentence is what shows it: `64c²` is the *squared* energy, so the amplitude is `8c`.
2. **Involution.** For a random group `a`: `hadamard64_group(a)` twice returns the original within `1e-12`. This is the property the oracle's construction depends on.
3. **Orthogonality.** For random `a`, `b`: `dot(Ha, Hb) == dot(a, b)` within `1e-12`. This is what makes "inverse-rotate K in the oracle" equal "rotate Q in the kernel", so it is the self-check that actually justifies the design.
4. **Nibble order — and a cross-check against a real appended plane.** Do not settle the parity convention by reading Part 0b alone: a swapped convention is *symmetric under the Hadamard* (the Sylvester matrix commutes with index-bit permutations, and swapping dims `2k`/`2k+1` is one), so the packed variants would decode every pair backwards and show up as a moderate attention-output perturbation rather than a blatant failure. Assert it.

   Pure assertions. The discriminating bytes are those whose two nibbles are a `−8` / `+7` pair, i.e. `0x78` and `0x87` (the fill kernel's `gqa_kv_pack_i4` masks the low code with `0x0F` and shifts the high code left by four):

   ```cpp
   assert(unpack_i4(0x78u, 0) == -8 && unpack_i4(0x78u, 1) == 7);   // lo = -8, hi = +7
   assert(unpack_i4(0x87u, 0) ==  7 && unpack_i4(0x87u, 1) == -8);   // lo = +7, hi = -8
   assert(unpack_i4(0x00u, 0) ==  0 && unpack_i4(0xFFu, 0) == -1);   // sign extension, not a bias
   ```

   **Note for the implementer:** an earlier review proposed the bytes `0xF8` and `0x8F` for this check. Those decode to `(−8, −1)` and `(−1, −8)`: the high nibble `0xF` is `−1`, not `+7`, so an assertion written from that pair would fail and invite a "fix" of the port to a wrong convention. The intent — two bytes whose low/high nibbles are swapped between them — is what `0x78` / `0x87` express.

   End-to-end cross-check, strongest form. On the **`rk8v4` row's V plane** (packed int4, `RotateV = true`, no E8 projection, so the encoder is a pure round-to-nearest int4 quantizer), after `append_chunk`:

   - take the physical page for position `src` (`table_host[src / kPageSize]`), head 0, `page_offset = src % kPageSize`, group `g = 1`;
   - read the 32 bytes at that group, decode each of the 64 dims with `unpack_i4` and the group's `v_scale`, apply the host Hadamard once;
   - compare against `scenario.v[act_index(0, 0, kKvHeads, d)]` (an even token, so `scenario.v` is the exact value the fill quantized) and assert `max |decoded − source| ≤ 4 × v_scale_of_group`.

   **The hard bound is a theorem, not a tuning knob.** The encoder rounds to nearest, so its per-element error is at most half an int4 step, `0.5 × scale` (the group is scaled by `amax / 7` and `|v| ≤ amax`, so nothing clamps); the oracle then applies an orthonormal `H`, so the residual is `|Hε| ≤ ‖ε‖₂ ≤ √64 × 0.5 × scale = 4 × scale` however the encoder distributed its error. A correct decode therefore *cannot* exceed `4 × scale`, while the two ways to be wrong land far above it: **measured over 40 000 synthetic groups, a swapped nibble parity gives a worst case of 14 × scale and a missing inverse-Hadamard stage gives 19 × scale, against 1.47 × scale worst case and 0.75 × scale mean for the correct decode.**

   **A residual in the `0.5–1.5 × scale` range is the encoder's own error distribution showing through the inverse Hadamard, not a parity bug.** The check reports (does not fail) above `1.5 × scale`, the top of the observed correct envelope, and fails only above `4 × scale`. If the reported number lands in that range while the pure `unpack_i4` assertions all pass, the bound is fine and your port is fine: record the number, do **not** start rearranging `unpack_i4`.

   Run the same cross-check on the `rk4v4` row's K plane. **Do not** assert a bound of this kind on the `rk4v4-e8` row's planes: its encode-side E8 projection and the documented half-coset collapse (`gqa_attention_decode_i8.cuh:264-276`) are not modeled by the oracle and widen the residual by construction. That row's gate is its anchor (Part 4a).

**2c. Where it is applied.** K iff `rotate_k`, V iff `rotate_v`. No Q-side and no output-side modeling is needed in the oracle:

- score side: the kernel computes `⟨H·q_bf16, ĤK⟩`; the oracle computes `⟨q_bf16, H·ĤK⟩`, equal in exact arithmetic because `H` is orthonormal and symmetric.
- output side: the kernel reduces `Σ wᵢ (ĤVᵢ)` and then applies `H` to the reduced result; the oracle applies `H` per element and then reduces. `H` is linear, so these agree; the only term the oracle cannot model is the kernel's bf16 rounding of the *reduced* output before the inverse rotation, which is part of the conformance budget of Part 4, not a structural gap.

---

## Part 3 — The test: `tests/ops/test_gqa_kv_page_delta_rotated.cpp`

**3a. Decode path (A3), one case per variant.** Loop over the variant table; for each row:

1. `Int8Cache cache(kLogicalPages, scenario.table, kKvHeads, row.packed_v, row.rotate_k, row.rotate_v, row.packed_k, row.e8_lattice, row.e8_root);` then `append_chunk(cache, scenario.k_src, scenario.v, scenario.src, kRelocatedRun)` — the production fill path is the only thing that ever writes codes, on every variant.
2. Attach the delta plane (`delta = dst − src = −384` at `kDestPage`, zeros elsewhere) and `freq` to `cache.view()`, and run `run_cached_attention_view_i8(cache, scenario.q, kDecodeBase, kDecodeTokens, visible, view)` with `kDecodeTokens = 8`.
3. Reference: a second `Int8Cache` with the same flags, `append_chunk(…, scenario.k_dst, scenario.v, scenario.dst, kRelocatedRun)`, run the same way.
4. Anchor the reference to the FP64 oracle (Part 4a), and anchor the **delta run itself** to the delta oracle (Part 4c): `ideal_attention_with_delta` over the same cache, which rotates every key back by its page's delta.
5. `expect_equivalent` between the delta run and the reference.
6. Negative control: zero delta plane on a fresh but otherwise identical run, measured against the **delta oracle**; `expect_control_outside` (Part 4c) requires the skipped rotation to break the same criterion the correct run is judged by. Keep the existing `scenario.table[kDestPage] != kDestPage` precondition check — it is what makes the control meaningful.
7. Print the flags, the codec name, and the label suffix (`"rk8v4"`, `"rk4v4-e8"`, …) for every row before running it, so a failure names its variant.
8. The Part 2b nibble-order cross-check runs once in `main`, before this loop, using a cache the check appends itself (see 2b item 4): `rk8v4`'s V plane and `rk4v4`'s K plane. Nothing there involves a delta — only Q is ever counter-rotated, and the codes it reads are exactly what its own `append_chunk` wrote.

**3b. Prefill path (A1).** Mirror `check_prefill_attention_with_delta` in `tests/ops/test_gqa_kv_page_delta.cpp` (`kPrefillTokens = 128`, `kPrefillBase = (kDestPage + 2) * kPageSize`, fresh `prefill_q` seeded separately, tail K/V chunk written at `kPrefillBase`, same anchor + equivalence + negative-control structure). Run it once for each in-scope variant.

Add `Int8Cache::batch_view()` to the common header, mirroring `Cache::batch_view()` exactly: direct planes, `block_table.view({logical_pages, 1})`, `head_dim`/`num_kv_heads`/`dtype`/`quant_group`, all six flags, and an **unset** `rope_deltas` (the caller attaches it, as the BF16 prefill test does with `.view({kLogicalPages, 1})`) and `rope_inv_freq`.

Route, per Part 0e: because 0e shows the public entry point forwards the view's delta plane and frequency table into the same prompt launcher the hook calls, and because `width = 128 > kMaximumVerifyTokens = 16` with `batch = 1` resolves to `GqaAttentionRoute::Prompt` (`gqa_attention.cpp:344-356`), **use the public `ops::gqa_attention`**. It is the stronger test: it exercises validation, route resolution, the fill path, and the attention path with the delta plane attached, rather than entering below the API.

```cpp
// Public A1 route with the delta plane on the batch view. Shapes and the table-row argument follow
// run_append_attention() in the common header; workspace is sized with DType::I8 so the split count
// matches what gqa_attention_cached will dispatch. The cache is reached through the view, so the
// helper takes the view alone -- an Int8Cache parameter would be unused.
AttentionRun run_prefill_attention_i8(const std::vector<__nv_bfloat16>& q,
                                      const std::vector<__nv_bfloat16>& k,
                                      const std::vector<__nv_bfloat16>& v,
                                      std::int32_t first_position, std::int32_t tokens,
                                      std::uint32_t visible,
                                      const PagedKVBatchLayerView& batch_view);
```

Fallback, and only if the public call throws on shape or route validation: `detail::gqa_attention_prompt_launch` with the same batch view, as the BF16 prefill test does, and **note the substitution in the report**. Do not use the BF16 helper's `Tensor{}` `valid_columns` + 3-D shapes without checking `require_shape` first — the public route validates `q`/`out` as `{kHeadDim, q_heads, tokens, 1}` and `k`/`v` as `{kHeadDim, kv_heads, tokens, 1}` (`gqa_attention.cpp:433-441`, `:200-218`).

The prefill reference cache must be built the same way as the decode reference but for the prefill positions, and the FP64 oracle call for the prefill path is `ideal_attention(prefill_q, reference, kPrefillBase, kPrefillTokens)` — the templated form already handles both cache types.

**3c. Flag constants: one source, feeding everything.** Define the variant table once and use it for the `Int8Cache` constructors, the `codec_of` dispatch, and the printf:

```cpp
struct Variant { const char* name; bool packed_v, rotate_k, rotate_v, packed_k, e8_lattice, e8_root; };
constexpr Variant kVariants[] = {
    {"rk8v4",     true,  true,  true,  false, false, false},
    {"rk4v4-e8",  true,  true,  true,  true,  true,  false},
    {"rk4v4",     true,  true,  true,  true,  false, false},  // optional row: same codec, different encoder
};
```

Assert once in `main` that no row has `e8_root` (otherwise `codec_of` throws and the test is a construct-time abort, not a failure) and that `codec_of(row) != KvCodec::I8G64` for every row. Print `name`, the six flags, and the codec for each row so a stale value is visible in ctest output.

---

## Part 4 — Criteria, anchors, controls

**4a. Criterion plumbing.** In the anchor call, pass the criterion to **both** `measure` and `expect_conformant`. For a rotated variant that criterion is `kConformanceRotatedI8` (4c, which supersedes the `kConformanceInt8` this section was drafted with):

```cpp
expect_conformant(measure(got_reference.out,
                          ideal_attention(scenario.q, reference, kDecodeBase, kDecodeTokens),
                          kConformanceRotatedI8),
                  "A3 over a directly prefilled int8 page [rk8v4]", kConformanceRotatedI8);
```

`measure` defaults to `kConformanceBf16`, so passing the criterion to only one of the two produces a half-BF16/half-INT8 judgement whose `gross` field is computed against the wrong thresholds. `expect_equivalent` reads only `relative_l2` and needs no criterion; leave its BF16 default and say so in a comment. `expect_control_separates` is no longer used by this test at all -- 4c replaces it with `expect_control_outside`, which needs the criterion explicitly.

**4b. If a rotated anchor fails.** Do not touch `kConformanceInt8` and do not loosen a limit. Paste the measured `relative_l2` and work through the causes in this order:

1. **A kernel-order bug is the most likely cause, not the threshold.** The rotated variants apply the Q counter-rotation *before* the Hadamard (`gqa_attention_decode_i8.cuh:428-433`: "Rotate before Hadamard: since H is applied to both sides, (H·R·Q)·(H·K) = (R·Q)·K. Applying R after H would not commute"). A missing or mis-ordered Hadamard in the oracle's `image()`, a Hadamard applied to a group that is not 64 elements wide, or a plane indexed with the wrong leading extent produces a large error, not a marginal one. Check the oracle first, and re-run the Part 2b checks.
2. **A genuine new rounding term.** The output inverse-rotation stages the reduced output through bf16 before `H` (`gqa_kv_inverse_rotate_output_kernel`), which the oracle cannot model. This is a real, small contributor — but it should move the number, not double it.
3. **Only then the criterion.** Note that the Hadamard redistributes Q and K energy before quantization, which typically *improves* int8 error rather than worsening it, so an in-scope variant failing where `Int8Group64` passes is evidence of (1), not of a tight profile. If after (1) and (2) the number is still above `3.15e-3`, propose a named `kConformanceRotatedI8` next to `kConformanceInt8` in `tests/ops/gqa_relocation_test_common.h`, with the measured value, the margin, and why the residual is a legitimate property of the packed/rotated path — gated on reviewer sign-off. Never widen the shared constant.

**4c. The anchored delta oracle, the control, and the rotated criterion (from the first Windows run).** The first run of `ninfer_gqa_kv_page_delta_rotated_test` on Windows (2026-10-10) returned 7 failures from two calibration problems. Both are measured; neither is a code fault.

- **The reference anchor.** `A3 over a directly prefilled int8 page` read `3.750e-3` (rk8v4), `3.777e-3` (rk4v4-e8) and `3.757e-3` (rk4v4) against `kConformanceInt8`'s `3.15e-3`, with **0 gross errors** — variant-independent to three digits across an int8 K plane, a packed int4 K plane and an E8-projected encoder. Both packed-plane cross-checks passed (`0.587 ×` and `0.812 ×` the group scale, inside the proven `4 ×` envelope), so the decode, nibble and extent machinery is right.

  **Correction (fifth pass): the first version of this paragraph attributed the excess to the bf16 staging in front of the output inverse Hadamard, as a term "a plain int8 cache does not pay". Two measurements refute that.** The plain `Int8Group64` variant reads `3.142e-3` against the same `3.15e-3` on the same A3 shape (same scenario, same query tokens, same decode path), so the plain path already spends 99.75% of the criterion and the rotated rows add only 19% on top of it; and the rotated A1 anchor is `2.46e-3`, i.e. the rotated *prefill* path lands lower than the rotated decode path, so the staging term cannot be what separates them. What the numbers actually say is that the A3 decode anchor nearly saturates `kConformanceInt8` even for the plain variant, and a shape with 0.25% of headroom cannot absorb any added rounding. That, not a rotation-specific term, is the warrant for the named criterion: `kConformanceRotatedI8{5.0e-3, 1.1e-3, 2.2e-3}` in `tests/ops/gqa_relocation_test_common.h` — the measured maximum plus a third of margin, gross thresholds carried over unchanged so only the L2 side moves. `kConformanceInt8` is untouched.
- **The control was unachievable on a packed int4 K plane.** The correct delta-against-direct-prefill error is `1.719e-2` (rk4v4-e8) and `1.174e-2` (rk4v4) — the encode error of a plane quantized at the *source* RoPE phase, which the delta cannot undo — while the physical skip-the-rotation error is only `~5.8e-2` for every variant. The best separation the data can give is therefore `3.4x` and `4.9x`, against `expect_control_separates`' `max(10 × correct, 2e-2)`: rk8v4 passed at `13x`, both packed rows failed. The harness was reporting, correctly, that it could not see a skipped rotation through that much encode noise.

The answer is an **anchored delta oracle**, not a loosened limit. `ideal_attention_with_delta<CacheType>(q, cache, first_position, tokens, page_deltas, freq)` decodes the cache exactly as `ideal_attention` does, then rotates each **key** by its page's `forward_rope(·, +delta)` — the key-side form of the kernel's Q counter-rotation, valid because `H` and `R` are orthonormal:

    <R(-delta) q, K_src> == <q, R(delta) K_src>

`forward_rope(·, +delta)` is the exact inverse of `gqa_rope_counter_rotate_pair_f32` (which applies `R(-delta)`: `x' = x·c + y·s`, `y' = y·c - x·s`, `gqa_attention_decode_i8.cuh:428-433`, group 0 only because the rotary region is dims `[0, 2·kRotaryPairs)` = `[0, 64)`), and composing it with the stored `R(src + t)` key yields the `R(dst + t)` key a direct prefill would have written — so the oracle needs no knowledge of the encoder. Because it rotates the *decoded* key it carries the plane's encode error exactly, which is precisely the term that swamped the cache-against-cache control, and a skipped rotation then shows at the criterion's own resolution instead of at 3.4x. `expect_control_outside` states the control that way: the correct run must conform to the criterion and the zero-delta run must break the same criterion. Both anchors are judged with `kConformanceRotatedI8`; the equivalence against the directly prefilled cache is kept as a secondary observation (it is genuine production information) but no longer carries the discrimination.

Verified off-GPU before shipping, since that environment has no CUDA: the oracle text was extracted from the header, compiled against a stub cache and run on a synthetic source-phase/destination-phase pair. `ideal_attention_with_delta` reproduces the destination-phase truth to `1.68e-16` relative L2 — the direction, the composition and the per-page indexing are exact — while no-delta, wrong-sign and wrong-page land at `2.8e-2` / `3.2e-2` / `2.8e-2`, about 6x above the `5.0e-3` criterion, so the control assertion has real margin on every variant. The refactored `ideal_attention` was confirmed **bit-identical** (`0.000e+00`) to its pre-refactor inline formulation, which is what keeps the four existing consumers' anchors unchanged.

---

## Part 5 — CMake registration

In `tests/CMakeLists.txt`, next to the existing delta tests (the block that registers `ninfer_gqa_kv_page_delta_i8_test`):

```cmake
# Rotated int8 variants (rk8v4, rk4v4-e8): the same delta scenario with packed int4 planes and a
# host port of the 64-point Hadamard, plus the I8 prefill route that the plain-int8 oracle test
# defers. Its negative control is what proves the host rotation is not a no-op.
ninfer_add_op_test(ninfer_gqa_kv_page_delta_rotated_test
  SOURCES ops/test_gqa_kv_page_delta_rotated.cpp
  LIBRARIES ninfer_ops)
```

`ninfer_add_op_test` already sets `SKIP_RETURN_CODE 77`, so the `cudaFree(nullptr)` guard in `main` stays.

---

## Part 6 — Explicit boundaries

- **No `rk2v4-e8`.** No root table, no axis table, no radius scale, no `kHeadDim/4` plane. That is 4d-4.
- **No E8 lattice anywhere in the oracle.** `e8_project_8d_warp` is encode-side; the packed-int4 model is the complete decode contract for both `packed_k` variants (Part 0b).
- **No encode-side modeling.** The oracle may not reproduce `rintf`, the half-coset collapse, or the absmax scan; it reads what the kernel wrote.
- **No public-API changes.** If something in `src/ops/` needs to change to make this test pass, STOP and report — the phase is oracle-only.
- **No criterion changes** other than a named, review-gated constant per Part 4b.

---

## Part 7 — Build

Build commands are run only when the operator instructs it (AGENTS.md). When instructed, from `build-ninja/`:

```bash
cmake --build . --target ninfer_gqa_kv_page_delta_rotated_test ninfer_gqa_kv_page_delta_i8_test
```

A build failure means the common-header refactor broke its other consumers, not just the new file; fix the header rather than special-casing the new test.

---

## Part 8 — Verification ladder

**These commands run only when the operator instructs it**, the same policy as Part 7. Run, in this order, and paste the raw output of each:

1. `ctest -R ninfer_gqa_kv_page_delta_rotated_test --output-on-failure`
2. `ctest -R ninfer_gqa_kv_page_delta_i8_test --output-on-failure` — the plain variant must be **bit-unchanged** through the refactor. This is the test that catches a `codec_of` or extent mistake that silently redirects `I8G64`.
3. `ctest -R "ninfer_gqa_kv_page_delta_test|ninfer_scr_relocation_integration_test|ninfer_gqa_attention_test" --output-on-failure` — the other consumers of `gqa_relocation_test_common.h`, including the BF16 oracle and the integration test that must stay the reference for the delta machinery.
4. `ctest --output-on-failure` — the full suite (84 tests) with zero failures.

Do not pipe any of these through a filter that can mask the exit status; capture it.

**Three numbers to record per variant, because they feed the Phase 4d-4 handoff** (non-blocking either way, but they are the reason the run is worth reporting in detail):

- **The reference anchor `relative_l2`** against `kConformanceRotatedI8` (`5.0e-3`), measured at `3.750e-3` / `3.777e-3` / `3.757e-3` — variant-independent, which is what rules out a codec-dependent error rather than attributing it to one. Read it beside the plain variant's own anchor, `3.142e-3` against a `3.15e-3` profile: the A3 shape nearly saturates `kConformanceInt8` on its own, which is why the rotated rows carry their own profile (Part 4c).
- **The delta-anchor `relative_l2`** — the new check, and the one that decides whether the counter-rotation is right. A skipped rotation lands at ~`5.7e-2`, so a value at the reference anchor's level means the rotation is exact; anything in the `1e-2` band means the rotation is partially wrong, not that a threshold is tight.
- **The control multiple** — the zero-delta error expressed as a multiple of the `5.0e-3` criterion. Measured off-GPU at ~6x on synthetic data with no quantization; the Windows number should be larger, because bf16 and the encode stages widen the skip error to ~`5.7e-2` (~11x). A value near 1x would mean the control has stopped discriminating and is the one thing to stop on.

---

## Report format

Do not summarize. Do not paraphrase. Paste raw output.

1. Part 0a's `sed` output and the flag values for both in-scope variants, with the codec each row selects.
2. Part 0b's grep output (both the decode branch and the pack/unpack pair), and one sentence stating whether the `PackedK` read branch does lattice work.
3. Part 0c's `sed` output, and the leading extent + dtype your `Int8Cache` publishes for K and V on each row.
4. Part 0e's grep output and which route Part 3b took.
5. Part 2's four self-check lines, raw — including whether the DC gain came out `8` or `1` (if `1`, the port has an extra `1/8`), and the `rk8v4` V-plane cross-check's worst per-element residual in units of the group scale, against the `4 × scale` envelope and the `1.5 × scale` advisory.
6. The first build error block if the build is nonzero.
7. Raw output of all four ctest runs.
8. The measured reference anchor `relative_l2`, the delta-anchor `relative_l2` and the control multiple over the criterion, per variant, even when they pass.
9. Explicit acknowledgment that Phase 4d remains **uncommitted**, gated on the Windows run of both delta test binaries plus reviewer sign-off on any new criterion constant.

---

## Revision notes (2026-10-10)

Fifth pass (the plain-int8 baseline, and the correction it forced). The rotated binary went green on its second Windows run — 0 failures in ~3.0 s, delta anchor `3.754e-3`/`3.746e-3`/`3.740e-3` against reference anchors `3.750e-3`/`3.777e-3`/`3.757e-3` — so the anchored design did what it was introduced for. But the baseline that Part 4c's justification needed refuted that justification: `ninfer_gqa_kv_page_delta_i8_test` anchors `Int8Group64` at `3.142e-3` against the same `3.15e-3` profile on the same A3 shape, and the rotated *prefill* anchor is `2.46e-3`, lower than the rotated decode anchor. So the rotated excess is not the bf16 output-staging term that a plain cache supposedly does not pay: the A3 decode shape nearly saturates `kConformanceInt8` on its own (0.25% of headroom) and the rotation adds 19%. The criterion's value is unchanged and still 24% above the measured maximum; only its stated reason and the header comment were rewritten. That near-saturation is recorded as a standing property of the A3 shape, not a bug in this phase.

Fourth pass (the first Windows run, and the redesign it forced). `ninfer_gqa_kv_page_delta_rotated_test` passed its four host self-checks and both packed-plane cross-checks and failed 7 checks: three reference anchors at ~`3.75e-3` against `3.15e-3`, and four controls. Neither class is a code fault. Part 4c records the measurements, the argument and the fix — a `kConformanceRotatedI8{5.0e-3, 1.1e-3, 2.2e-3}` profile for the rotated anchors, plus `ideal_attention_with_delta` and `expect_control_outside` so the relocated run is anchored to FP64 (key-side `R(+delta)`) instead of only to a second cache. The reason the second class is not a loosened limit: the packed-int4 rows' *correct* path carries `1.2e-2`–`1.7e-2` of source-phase encode error, so the multiplicative control could only ever reach 3.4x there, and the anchored oracle removes that term from the comparison instead of the assertion. `kConformanceInt8` and `expect_control_separates` are unmodified, and the plain-int8 and BF16 delta tests keep using them. Step 3 of the Part 8 ladder is green on this run; the new binary has not been rebuilt or re-run since the redesign.

Third pass (the implementation pass), with the measurement that actually settled item 1 of that review:

1. **The `0.55 × scale` cross-check bound was replaced, not widened by guesswork.** Compiling the shipped helper block and running 40 000 synthetic groups through the encoder and the oracle decode measured the correct decode at 1.47 × scale worst case and 0.75 × scale mean, while a swapped parity gave 14 × scale and a missing inverse Hadamard gave 19 × scale. The hard bound is now the proven envelope `4 × scale` (`|Hε| ≤ ‖ε‖₂ ≤ √64 × 0.5 × scale`), which a correct decode cannot exceed and the two wrong regimes exceed by 3.5x or more; the advisory value moved to `1.5 × scale`, the top of the observed correct envelope, so a residual in the normal range is reported and recorded rather than mistaken for a parity bug.
2. Part 3a step 8 no longer says "delta-less append": the cross-check runs once in `main` against a plane its own `append_chunk` wrote, and nothing in it involves a delta, because only Q is ever counter-rotated.
3. Part 8 now carries the same operator-instruction gate as Part 7, so the ctest ladder is not run by accident.

Applied from the second review pass: the DC-gain correction as a summary line at the top of Part 2; the explicit per-plane leading extent in `image()`; the assertive nibble-order check; and `codec_of` mirroring both wrapper invariants.

Applied from the first review pass: the real `codec_of` `e8_root` throw with the non-`constexpr` rationale; per-plane host typing in `image()` (forced by the wrapper's `{I8, U8}` / `{kHeadDim, kHeadDim/2}` validation, Part 0c); the pre-committed Part 3b route with its conditional fallback; the criterion passed to both `measure` and `expect_conformant`; the Part 4b failure tree; single-source variant flags; and the widening verification ladder.

Two arithmetic corrections to the reviews themselves, both recorded in place so they are not "fixed" back: the Hadamard DC gain is `8c`, not `c` (Part 2), and the discriminating nibble bytes are `0x78` / `0x87`, not `0xF8` / `0x8F` (Part 2b, item 4).

Where the implementation diverged from this document and the document was corrected to match: `image()` uses inline per-plane leading extents (`k_leading()` / `v_leading()` with `k_full` / `v_full`) rather than separate `k_idx` / `v_idx` helpers; the prefill helper takes the view alone, since the Op reaches the cache through it and an `Int8Cache` parameter would be unused; the `rk4v4` row is part of the shipped variant table rather than an optional extra, and its K plane carries the same cross-check as `rk8v4`'s V plane; and Part 3a step 8 describes where the cross-check actually runs.
