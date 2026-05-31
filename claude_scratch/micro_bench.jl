# Microbenchmark the rotation kernel's combine loop in isolation, comparing
# four implementations to pick the SIMD-friendliest.
using Pkg
Pkg.activate(joinpath(@__DIR__, "bench_env"))
using BenchmarkTools, Random

# Realistic L5I shapes at fs = 12 MHz.
const N_doppler = 150
const N_coh     = 10
const N_pat     = 10
const N_col     = 12_000

# A: current — outer q, middle p, inner ω contiguous; ComplexF32 MACs
function combine_A!(noncoh_buf, col_buf, sub_block_ffts, tiled, num_sign_patterns)
    num_doppler, num_coh = size(sub_block_ffts)
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf, zero(ComplexF32))
            for p in 0:num_coh-1
                @inbounds @simd for ω in 1:num_doppler
                    col_buf[ω] += tiled[ω, p + 1, q] * sub_block_ffts[ω, p + 1]
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = abs2(col_buf[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

# B: like A, but reinterpret the per-pattern slice as Float32 to drive SIMD packs
function combine_B!(noncoh_buf, col_buf, sub_block_ffts, tiled, num_sign_patterns)
    num_doppler, num_coh = size(sub_block_ffts)
    sb_re = reinterpret(reshape, Float32, sub_block_ffts)    # (2, num_doppler, num_coh) — (re, im)
    cb_re = reinterpret(reshape, Float32, col_buf)            # (2, num_doppler)
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf, zero(ComplexF32))
            for p in 0:num_coh-1
                tile_view = view(tiled, :, p + 1, q)            # contiguous ComplexF32 length num_doppler
                tile_re   = reinterpret(reshape, Float32, tile_view)  # (2, num_doppler)
                @inbounds @simd for ω in 1:num_doppler
                    ar, ai = tile_re[1, ω], tile_re[2, ω]
                    br, bi = sb_re[1, ω, p + 1], sb_re[2, ω, p + 1]
                    cb_re[1, ω] = muladd(ar, br, muladd(-ai, bi, cb_re[1, ω]))
                    cb_re[2, ω] = muladd(ar, bi, muladd( ai, br, cb_re[2, ω]))
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = abs2(col_buf[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

# C: split-real-imag stores for tiled and sub_block_ffts entirely
function combine_C!(noncoh_buf, col_buf_re, col_buf_im,
                    sb_re, sb_im, tiled_re, tiled_im,
                    num_sign_patterns, num_doppler, num_coh)
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf_re, 0f0); fill!(col_buf_im, 0f0)
            for p in 1:num_coh
                @inbounds @simd for ω in 1:num_doppler
                    ar = tiled_re[ω, p, q]
                    ai = tiled_im[ω, p, q]
                    br = sb_re[ω, p]
                    bi = sb_im[ω, p]
                    col_buf_re[ω] = muladd(ar, br, muladd(-ai, bi, col_buf_re[ω]))
                    col_buf_im[ω] = muladd(ar, bi, muladd( ai, br, col_buf_im[ω]))
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = muladd(col_buf_re[ω], col_buf_re[ω], col_buf_im[ω] * col_buf_im[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

# Baseline: original ±1 kernel for reference (Float32 sign)
function combine_baseline!(noncoh_buf, col_buf, sub_block_ffts, patterns, num_sign_patterns)
    num_doppler, num_coh = size(sub_block_ffts)
    for col_idx in 1:N_col
        for p in 1:num_sign_patterns
            fill!(col_buf, zero(ComplexF32))
            for k in 0:num_coh-1
                sign = patterns[k + 1, p]
                col_buf .+= sign .* view(sub_block_ffts, :, k + 1)
            end
            @inbounds for ω in 1:num_doppler
                cell_power = abs2(col_buf[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

Random.seed!(0)
sub_block_ffts = ComplexF32.(randn(ComplexF64, N_doppler, N_coh))
tiled = ComplexF32.(randn(ComplexF64, N_doppler, N_coh, N_pat))
patterns_pm1 = Float32.(rand((-1f0, 1f0), N_coh, N_pat))
col_buf = zeros(ComplexF32, N_doppler)
noncoh_buf = zeros(Float32, N_doppler, N_col)

# Split-real-imag prep
sb_re   = Float32.(real.(sub_block_ffts)); sb_im   = Float32.(imag.(sub_block_ffts))
tiled_re = Float32.(real.(tiled));         tiled_im = Float32.(imag.(tiled))
cb_re   = zeros(Float32, N_doppler);       cb_im   = zeros(Float32, N_doppler)

println("Combine loop microbench (N_col=$N_col, N_doppler=$N_doppler, N_coh=$N_coh, N_pat=$N_pat):")

t = @benchmark combine_baseline!($noncoh_buf, $col_buf, $sub_block_ffts, $patterns_pm1, N_pat) samples=10 seconds=10
println("  baseline (±1, ComplexF32): median = $(round(median(t).time / 1e6, digits=2)) ms")

t = @benchmark combine_A!($noncoh_buf, $col_buf, $sub_block_ffts, $tiled, N_pat) samples=10 seconds=10
println("  combine_A (Complex MAC):   median = $(round(median(t).time / 1e6, digits=2)) ms")

t = @benchmark combine_B!($noncoh_buf, $col_buf, $sub_block_ffts, $tiled, N_pat) samples=10 seconds=10
println("  combine_B (reinterpret):   median = $(round(median(t).time / 1e6, digits=2)) ms")

t = @benchmark combine_C!($noncoh_buf, $cb_re, $cb_im, $sb_re, $sb_im, $tiled_re, $tiled_im, N_pat, N_doppler, N_coh) samples=10 seconds=10
println("  combine_C (split storage): median = $(round(median(t).time / 1e6, digits=2)) ms")

# D: split tiled storage but reinterpret sub_block_ffts (ComplexF32) on the fly
function combine_D!(noncoh_buf, col_buf_re, col_buf_im,
                    sub_block_ffts_cf::Matrix{ComplexF32}, tiled_re, tiled_im,
                    num_sign_patterns, num_doppler, num_coh)
    sb = reinterpret(reshape, Float32, sub_block_ffts_cf)   # (2, num_doppler, num_coh)
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf_re, 0f0); fill!(col_buf_im, 0f0)
            for p in 1:num_coh
                @inbounds @simd for ω in 1:num_doppler
                    ar = tiled_re[ω, p, q]
                    ai = tiled_im[ω, p, q]
                    br = sb[1, ω, p]
                    bi = sb[2, ω, p]
                    col_buf_re[ω] = muladd(ar, br, muladd(-ai, bi, col_buf_re[ω]))
                    col_buf_im[ω] = muladd(ar, bi, muladd( ai, br, col_buf_im[ω]))
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = muladd(col_buf_re[ω], col_buf_re[ω], col_buf_im[ω] * col_buf_im[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

t = @benchmark combine_D!($noncoh_buf, $cb_re, $cb_im, $sub_block_ffts, $tiled_re, $tiled_im, N_pat, N_doppler, N_coh) samples=10 seconds=10
println("  combine_D (split tile, reinterpret sb): median = $(round(median(t).time / 1e6, digits=2)) ms")

# E: split tile (re/im) but read sub_block_ffts as ComplexF32 with real/imag() access
function combine_E!(noncoh_buf, col_buf_re, col_buf_im,
                    sub_block_ffts::Matrix{ComplexF32}, tiled_re, tiled_im,
                    num_sign_patterns, num_doppler, num_coh)
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf_re, 0f0); fill!(col_buf_im, 0f0)
            for p in 1:num_coh
                @inbounds @simd for ω in 1:num_doppler
                    ar = tiled_re[ω, p, q]
                    ai = tiled_im[ω, p, q]
                    b  = sub_block_ffts[ω, p]
                    br, bi = reim(b)
                    col_buf_re[ω] = muladd(ar, br, muladd(-ai, bi, col_buf_re[ω]))
                    col_buf_im[ω] = muladd(ar, bi, muladd( ai, br, col_buf_im[ω]))
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = muladd(col_buf_re[ω], col_buf_re[ω], col_buf_im[ω] * col_buf_im[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

t = @benchmark combine_E!($noncoh_buf, $cb_re, $cb_im, $sub_block_ffts, $tiled_re, $tiled_im, N_pat, N_doppler, N_coh) samples=10 seconds=10
println("  combine_E (split tile, reim sb):       median = $(round(median(t).time / 1e6, digits=2)) ms")

# F: split tile (re/im) PLUS sub_block_ffts as Float32 with stride-2 access via reinterpret-noreshape
function combine_F!(noncoh_buf, col_buf_re, col_buf_im,
                    sub_block_ffts::Matrix{ComplexF32}, tiled_re, tiled_im,
                    num_sign_patterns, num_doppler, num_coh)
    sb = reinterpret(Float32, sub_block_ffts)   # length 2*num_doppler*num_coh, layout same as ComplexF32 memory
    for col_idx in 1:N_col
        for q in 1:num_sign_patterns
            fill!(col_buf_re, 0f0); fill!(col_buf_im, 0f0)
            for p in 1:num_coh
                base = 2 * num_doppler * (p - 1)
                @inbounds @simd for ω in 1:num_doppler
                    ar = tiled_re[ω, p, q]
                    ai = tiled_im[ω, p, q]
                    br = sb[base + 2ω - 1]
                    bi = sb[base + 2ω]
                    col_buf_re[ω] = muladd(ar, br, muladd(-ai, bi, col_buf_re[ω]))
                    col_buf_im[ω] = muladd(ar, bi, muladd( ai, br, col_buf_im[ω]))
                end
            end
            @inbounds for ω in 1:num_doppler
                cell_power = muladd(col_buf_re[ω], col_buf_re[ω], col_buf_im[ω] * col_buf_im[ω])
                if cell_power > noncoh_buf[ω, col_idx]
                    noncoh_buf[ω, col_idx] = cell_power
                end
            end
        end
    end
end

t = @benchmark combine_F!($noncoh_buf, $cb_re, $cb_im, $sub_block_ffts, $tiled_re, $tiled_im, N_pat, N_doppler, N_coh) samples=10 seconds=10
println("  combine_F (stride-2 reinterpret):      median = $(round(median(t).time / 1e6, digits=2)) ms")
