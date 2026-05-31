# Rotation-search sensitivity: closing the ~5 dB gap to LongL5I

Status: implemented on `acquisition-secondary-code-3` (built on top of
`feat/secondary-code-rotation-v2` + the WIP per-column shift fix from
`/workspace/Acquisition`). All 16,571 tests pass.

## The gap

After the per-column shift fix landed (`mid-primary-code-shift-fix.md`), the
GPSL5I rotation search still trailed `LongL5I` by **~5 dB** of effective
sensitivity at CN0=45 dB-Hz:

| | Pre-fix GPSL5I+rotation | LongL5I |
|---|---|---|
| % not detected @ CN0=45 | 26.5% | 0% |
| Median peak/N @ CN0=45 | 27.8 | 123.7 |

## Root cause (two compounding effects)

**(A) Off-grid Doppler combination loss (3–8 dB).** The kernel summed
`Σ_p ±1 · sub_block_FFT_p[ω]` with ±1 NH10 cyclic-shift patterns. That's
matched-filter optimal only when the true Doppler lands on the coarse
`1/T_code = 1 kHz` sub-block FFT grid — between grid points the inter-sub-
block phase rotates by `δ ∈ (0, 2π)` per code period that ±1 cannot fit. A
direct probe of peak power vs LongL5I showed:

```
Doppler 100Hz: −6.05 dB | 200Hz: −7.35 dB | 500Hz: −7.98 dB | 1000Hz: 0 dB
```

(`claude_scratch/probe_gap.jl`, noiseless signal, cp=0.)

**(B) Noise-floor inflation from cell-wise max over rotations (~3 dB).** The
kernel computed `|·|²` for each of the 10 NH10 cyclic-shift hypotheses and
stored only the max per (ω, cp) cell. That collapses 10 cells into 1; for
noise the per-cell distribution becomes max-of-10 exponential, mean
`H₁₀ ≈ 2.93 ×` higher than single-exponential. A direct diagnostic at the
worst-case cp=5115:

```
peak_rot / peak_long  =  -0.71 dB   (essentially equivalent)
median_rot / median_long = +5.75 dB   (5.75 dB noise inflation)
```

(`claude_scratch/diag_pn.jl`.)

LongL5I avoids both: a single length-`num_doppler_bins` column FFT
automatically handles the inter-sub-block phase ramp (A), and every
hypothesis writes to its own cell (B).

## The fix (two orthogonal changes)

### (1) Inter-sub-block phase ramp

A length-N column FFT decomposes exactly as
```
FFT_N(x)[ω] = Σ_p exp(-2πi·p·s/N_coh) · sub_block_FFT_p[ω],  s = ω mod N_coh
```
so the optimal NH10-rotation-aware combination is
```
combine_q[ω] = Σ_p NH10[(p+r) mod L] · exp(-2πi·p·s/N_coh) · sub_block_FFT_p[ω]
```
i.e. the `±1` NH10 chip becomes a **complex phasor** that compensates for
the inter-sub-block Doppler progression. Empirically closes (A) to within
FFTW jitter: noiseless peak ratio = 0.996 (−0.02 dB) at every test Doppler.

Storage: a new per-PRN tensor pre-computed at plan time, pre-tiled along
the ω axis and **split into Float32 real/imag arrays** to drive Float32
SIMD in the inner loop (≈3× faster than ComplexF32 storage with the same
MAC count). The split-storage form measured ~14% **faster** than the
original ±1 kernel.

### (2) Per-rotation cp slice (drop cell-wise max)

Expand the noncoherent-buffer cp axis from `samples_per_code` to
`samples_per_code × num_secondary_rotations`. Each rotation hypothesis
gets its own column slice; no cell-wise max. The buffer's total cell count
becomes equal to LongL5I's (1.8 M × 10 = 18 M = 150 × 120 000), so the
CFAR statistics match LongL5I directly.

Result extraction decodes the peak col back into `(cp_within, rotation_idx)`
via div/mod by `samples_per_code`. The public `code_phase` output is
unchanged (it's `cp_within`); `rotation_idx` is currently hidden.

Drift correction (`_apply_code_drift!`) is unchanged in concept but now
circshifts within each rotation block independently — drift is phase-walk
of the primary code within one rotation hypothesis, so it must not wrap
across rotation-block boundaries.

## Implementation

Three files, two new src functions, ~150 LOC net:

- `src/sign_patterns.jl`: new factories `combined_phase_patterns(...)`
  (returns the compact (N_coh, num_patterns, N_coh) phasor) and
  `tile_phase_patterns(...)` (expands the compact form along the ω axis
  for SIMD-friendly inner loops).
- `src/plan.jl`: new fields
  `samples_per_code_eff`, `tiled_phase_patterns_re/im_by_prn`, new scratch
  `combine_buf_re/im`. Buffers sized at `samples_per_code_eff` on the
  rotation path. Plan builds the tiled split-form patterns once at
  construction time.
- `src/noncoherent_integration.jl`:
  - `_sign_search_step_with_rotations!` rewritten: split-real/imag
    Float32 inner combine loop with the complex phasor phasors, writes
    per-pattern to its cp slice (no max).
  - `_apply_code_drift!` updated to per-rotation-block circshift.
  - Dispatcher passes the new phasor arrays + combine scratch.
- `src/acquire.jl`: `_extract_result!` decodes `peak col → cp_within +
  rotation_idx`; uses `cp_within` for the public code_phase. Parabolic-
  interpolation neighbours stay within the same rotation block.
- `test/sign_patterns.jl`: new test sets for `combined_phase_patterns`
  (shape, DFT structure, s=0 degenerate case, pattern-independence of the
  phase ramp) and `tile_phase_patterns` (per-ω fine-class lookup).
- `test/secondary_code_search.jl`: new testset
  `"L5I rotation search — matches LongL5I (full 10 ms FFT) on off-grid Doppler"`.
  Asserts on-grid peak parity (always held) and off-grid peak parity at
  100 Hz and 500 Hz Doppler against a local `_LongL5I` shim. Was planted
  as `@test_broken`; promoted to live `@test` once the fix landed.

## Sensitivity (CN0 sweep, 472 trials, 60 cps × 8 Dopplers)

`claude_scratch/long_code_sweep.jl`:

| CN0 (dB-Hz) | GPSL5I+fix (mine) | LongL5I | Δ peak/N |
|---|---|---|---|
| 40 | **9.1%** miss, p/N=37.8  | 11.4% miss, p/N=41.6  | **−0.42 dB** |
| 42 | **1.1%** miss, p/N=59.4  | 1.5% miss,  p/N=63.5  | −0.29 dB |
| 45 | **0.0%** miss, p/N=113.7 | 0.0% miss,  p/N=123.7 | −0.37 dB |
| 47 | 0.0% miss,  p/N=177.6 | 0.0% miss,  p/N=194.5 | −0.39 dB |
| 50 | 0.0% miss,  p/N=356.9 | 0.0% miss,  p/N=380.1 | −0.27 dB |
| 55 | 0.0% miss,  p/N=1085.7 | 0.0% miss,  p/N=1158.5 | −0.28 dB |

- **Within 0.42 dB of LongL5I at every CN0** (target was within ~1 dB).
- At CN0=40 and 42, GPSL5I+fix actually BEATS LongL5I on detection rate.
- The residual ~0.3–0.4 dB matches the documented chip-boundary residual
  from the per-column shift fix (`~0.4 dB` per `mid-primary-code-shift-fix.md`).
  Closing that further requires a per-column fractional time-shift (phase
  ramp on the column data before sub-block FFT) — option 2 in the doc's
  "Future work" list — and is out of scope here.

## Doppler axis (noiseless probe)

`claude_scratch/probe_gap.jl`, GPSL5I peak / LongL5I peak vs Doppler:

```
   0.0 Hz: 0.996 (-0.02 dB)         500.0 Hz: 0.996 (-0.02 dB)
 100.0 Hz: 0.996 (-0.02 dB)         600.0 Hz: 0.996 (-0.02 dB)
 200.0 Hz: 0.996 (-0.02 dB)         700.0 Hz: 0.996 (-0.02 dB)
 300.0 Hz: 0.996 (-0.02 dB)         800.0 Hz: 0.996 (-0.02 dB)
 400.0 Hz: 0.996 (-0.02 dB)        1000.0 Hz: 0.996 (-0.02 dB)
```

Was −6 dB at 100 Hz, −8 dB at 500 Hz, −3.5 dB at 3333 Hz before the
phase-ramp fix; now 0 dB at every Doppler. The 0.02 dB residual is FFTW
rounding.

## Runtime

`claude_scratch/bench_env/bench_l5i.jl` and `bench_long.jl`, single PRN,
L5I 10 ms coherent, fs = 12 MHz, BenchmarkTools 20 samples / 15 s budget:

| Configuration | Median |
|---|---|
| doc baseline (±1 kernel + WIP shift) | 117.9 ms |
| **GPSL5I + my fix (phase ramp + cp expansion)** | **146 ms** |
| LongL5I (N_coh=1, simple pilot) | 175 ms |

- **20 % slower than the original ±1 kernel**, with 5 dB more sensitivity.
- **17 % faster than LongL5I**, with the same sensitivity to within 0.4 dB.
- Allocations: 64 B / 2 allocs, unchanged.
- Memory: per-thread noncoherent buffer grows 7→72 MB on the rotation
  path. Same total footprint as LongL5I's noncoherent buffer.

The runtime breakdown (microbench in `claude_scratch/micro_bench.jl`,
combine loop only, 12 000 cols × 10 patterns × 10 sub-blocks × 150 ω):

```
±1 ComplexF32 kernel (baseline):              60 ms
Complex MAC tiled (strided sub_block_ffts):   180 ms       ← rejected
Complex MAC tiled (reim split tile):           54 ms       ← chosen
Fully-split (re+im float buffers):             51 ms       ← marginally faster
```

The chosen layout (split tile + `reim(sub_block_ffts[ω, p])`) drives
Float32 SIMD without changing the sub-block FFT storage, so the FFTW step
is unchanged.

## Test coverage

Beyond all existing tests, added:

1. `test/sign_patterns.jl` — `combined_phase_patterns` + `tile_phase_patterns`:
   shape, DFT-matrix structure, s=0 degenerate case, ramp factors
   pattern-independently, per-ω tile lookup. ~1 100 sub-cases.
2. `test/secondary_code_search.jl` — `"L5I rotation search — matches LongL5I
   (full 10 ms FFT) on off-grid Doppler"`. Asserts ratio > 0.95 at 100 Hz
   and 500 Hz Doppler (the off-coarse-grid worst cases). Includes a local
   `_LongL5I` shim mirroring `claude_scratch/long_code.jl`.

The pre-existing `cp=5115` regression (`@test_broken` originally, promoted
by the WIP shift fix) continues to pass.

## Limitations / future work

1. **Rotation_idx is hidden.** The kernel internally tracks which NH10
   cyclic shift won, but the public `AcquisitionResults` doesn't expose
   it. Could be added as an optional field if downstream tracking wants it.
2. **Residual 0.3–0.4 dB.** The remaining gap to LongL5I matches the
   chip-boundary residual the per-column shift fix could not eliminate
   (~½ a block of misalignment). Closing it requires a per-column
   fractional time-shift on the column data before sub-block FFT — option
   2 in the doc's "Future work" list. Not necessary to meet the
   "within ~1 dB" sensitivity target.
3. **N_nc > 1 + rotation drift on extreme Dopplers.** `_apply_code_drift!`
   now circshifts per-rotation-block; for typical drift values (< chip
   boundaries) this is correct. The existing N_nc=4 + 5000 Hz test still
   passes.

## Files

- `src/sign_patterns.jl` — `combined_phase_patterns`, `tile_phase_patterns`
- `src/plan.jl` — `samples_per_code_eff`, `tiled_phase_patterns_re/im_by_prn`, scratch `combine_buf_re/im`
- `src/noncoherent_integration.jl` — rewritten `_sign_search_step_with_rotations!`; per-block `_apply_code_drift!`
- `src/acquire.jl` — `_extract_result!` decodes `(cp_within, rotation_idx)`
- `test/sign_patterns.jl` — new test sets for phase-ramp factories
- `test/secondary_code_search.jl` — new algebraic-equivalence test
- `claude_scratch/probe_gap.jl` — noiseless Doppler-axis equivalence sweep
- `claude_scratch/diag_pn.jl` — diagnostic that isolated the noise-inflation root cause
- `claude_scratch/long_code_sweep.jl` — headline CN0 detection sweep vs LongL5I
- `claude_scratch/bench_env/bench_l5i.jl`, `bench_long.jl` — runtime benchmarks
- `claude_scratch/micro_bench.jl` — combine-loop layout comparison
