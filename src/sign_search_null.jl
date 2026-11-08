# src/sign_search_null.jl
#
# CFAR null distribution of a power surface whose cells are the MAX over several
# sign hypotheses (data-bit polarities, bit-edge alignments) — the surface every
# sign search produces. `cfar_threshold(pfa, num_cells; …)` assumes each cell is
# one exponential power (χ²(2)) per non-coherent step; a max over K correlated
# hypotheses has a higher mean, which the noise estimate absorbs, and a much
# lighter tail RELATIVE to that mean, so the Gamma threshold is far too high
# (measured: realised Pfa 0% at a nominal 10% for 5 bit-edge hypotheses).

"""
    SignSearchNull

Null (noise-only) distribution of the test statistic `peak_to_noise_ratio` for
one search cell when the cell's power is the maximum over `num_hypotheses`
correlated sign hypotheses, summed over `num_noncoherent_integrations` steps.
Built by `plan_acquire` and carried by every [`AcquisitionResults`](@ref) of
that plan; [`cfar_threshold`](@ref) reads its tail. Internal: tabulates
`log P(peak_to_noise_ratio > pnr)` on an increasing `pnr` grid.
"""
struct SignSearchNull
    num_hypotheses::Int
    pnr::Vector{Float64}
    log_survival::Vector{Float64}
end

# Per-cell threshold: the `pnr` at which the tabulated log-survival reaches
# `log(pfa_cell)`, interpolated linearly (log-survival is near-linear in pnr);
# past the table, extrapolated along its last segment.
function _threshold(null::SignSearchNull, pfa_cell::Float64)
    target = log(pfa_cell)
    pnr, ls = null.pnr, null.log_survival
    i = findfirst(<=(target), ls)
    if i === nothing
        i = length(ls)
    elseif i == 1
        return pnr[1]
    end
    slope = (ls[i] - ls[i-1]) / (pnr[i] - pnr[i-1])
    return pnr[i-1] + (target - ls[i-1]) / slope
end

# Per-row signs (`num_doppler_bins × K`) of the K sign hypotheses one cell is
# maximised over: rotation 0's data-bit polarities × bit-edge alignments, built
# with the kernels' own sign construction (`_fill_hypothesis_signs!`) for a
# column aligned to row 0. Rotations are not included — each rotation has its
# own cell (`get_num_cells`).
function _hypothesis_signs(patterns::Matrix{Float32}, num_data_bits::Int,
        num_coh::Int, num_edges::Int, num_blocks::Int)
    num_doppler_bins = num_coh * num_blocks
    num_data_combos = 1 << (num_data_bits - 1)
    signs = Matrix{Float32}(undef, num_doppler_bins, num_data_combos * num_edges)
    k = 0
    for data_combo in 0:num_data_combos-1, edge in 0:num_edges-1
        k += 1
        _fill_hypothesis_signs!(view(signs, :, k), patterns, data_combo + 1, edge, 0,
            num_data_bits, num_coh, num_edges, num_blocks)
    end
    return signs
end

"""
    _sign_search_null(signs, num_noncoherent_integrations; num_samples, rng) -> SignSearchNull

Null distribution of the per-cell statistic for the sign hypotheses `signs`
(rows × K, entries ±1).

Under noise, the K hypothesis outputs of one cell, `X_k = Σ_m s_k[m] x[m] e^{-iωm}`,
are jointly complex Gaussian with correlation `R = SᵀS / N` (white noise across
rows), so each power `|X_k|²` is unit exponential (in units of one hypothesis'
noise power) and the cell is `W = max_k |X_k|²`.

1. `P(W > w)` on a grid, by the ALOE rare-event estimator (Adler, Blanchet &
   Liu, 2012): draw a hypothesis `k` uniformly, draw `X` conditioned on
   `|X_k|² > w`, count the hypotheses `N(X)` above `w`; then
   `P(W > w) = K e^{-w} E[1/N]`. Bounded relative error at any tail depth, so
   the ~1e-8 per-cell false-alarm probabilities CFAR needs come out of a few
   thousand samples.
2. `M = num_noncoherent_integrations` steps sum `M` independent copies of `W`.
   For `M > 1` the tail of the sum comes from the Lugannani–Rice saddle-point
   approximation, with `W`'s cumulant generating function integrated from step 1.
3. The noise estimate is the surface mean, `M · E[W]`, so the statistic is
   `Y / (M · E[W])` — the table's `pnr` axis.

Deterministic: `rng` defaults to a fixed seed.
"""
function _sign_search_null(signs::AbstractMatrix, num_noncoherent_integrations::Int;
        num_samples::Int = 4000, rng = Xoshiro(0x5157))
    num_rows, K = size(signs)
    S = Float64.(signs)
    R = (S' * S) ./ num_rows
    # Z = A g ~ CN(0, R) for g ~ CN(0, I); R may be singular (repeated hypotheses).
    eig = eigen(Symmetric(R))
    A = eig.vectors * Diagonal(sqrt.(max.(eig.values, 0.0)))

    # Common random numbers across the w grid keep the survival curve smooth.
    Z = A * (randn(rng, ComplexF64, K, num_samples))
    ks = rand(rng, 1:K, num_samples)
    excess = randexp(rng, num_samples)
    phase = cis.(2π .* rand(rng, num_samples))

    dw = 0.25
    w_grid = collect(0.0:dw:60.0)
    log_sw = similar(w_grid)
    X = zeros(ComplexF64, K)
    for (i, w) in enumerate(w_grid)
        acc = 0.0
        for n in 1:num_samples
            k = ks[n]
            c = sqrt(w + excess[n]) * phase[n]
            δ = c - Z[k, n]
            count = 0
            @inbounds for j in 1:K
                count += abs2(Z[j, n] + R[j, k] * δ) >= w
            end
            acc += 1 / count
        end
        log_sw[i] = min(0.0, log(K) - w + log(acc / num_samples))
    end
    # E[W] = ∫ P(W > w) dw, exponential within each grid segment; beyond the
    # grid the tail is e^{-w}-like and contributes ≲ K·e^{-60}.
    mean_w = 0.0
    for i in 1:length(w_grid)-1
        a, b = log_sw[i], log_sw[i+1]
        mean_w += abs(a - b) < 1e-12 ? exp(a) * dw : dw * (exp(a) - exp(b)) / (a - b)
    end

    M = num_noncoherent_integrations
    if M == 1
        return SignSearchNull(K, w_grid ./ mean_w, log_sw)
    end
    pnr, log_s = _saddlepoint_sum_tail(w_grid, log_sw, M)
    return SignSearchNull(K, pnr ./ (M * mean_w), log_s)
end

# Tail of the sum of M iid copies of W (survival tabulated as `log_sw` on the
# uniform grid `w_grid`), by Lugannani–Rice. Returns (y, log P(Y > y)).
function _saddlepoint_sum_tail(w_grid, log_sw, M)
    # Fine grid with log-linear interpolation, then the e^{-w}-like tail
    # extrapolated along the last segment far enough for e^{θw}·S(w) to vanish.
    dw = w_grid[2] - w_grid[1]
    tail_rate = (log_sw[end-1] - log_sw[end]) / dw
    h = 0.02
    w = collect(0.0:h:w_grid[end] + 60 / (1 - 0.95))
    ls = map(w) do x
        if x <= w_grid[end]
            i = min(floor(Int, x / dw) + 1, length(w_grid) - 1)
            t = (x - w_grid[i]) / dw
            (1 - t) * log_sw[i] + t * log_sw[i+1]
        else
            log_sw[end] - tail_rate * (x - w_grid[end])
        end
    end
    # E[e^{θW}] = 1 + θ ∫ e^{θw} S(w) dw, and its θ-derivatives, by parts.
    function cgf(θ)
        m0 = m1 = m2 = 0.0
        @inbounds for i in eachindex(w)
            x = w[i]
            f = exp(θ * x + ls[i]) * ((i == 1 || i == length(w)) ? h / 2 : h)
            m0 += f
            m1 += (1 + θ * x) * f
            m2 += (2x + θ * x^2) * f
        end
        mgf = 1 + θ * m0
        d1 = m1 / mgf
        return log(mgf), d1, m2 / mgf - d1^2
    end
    θmax = 0.95 * min(tail_rate, 1.0)
    θs = range(0.02, θmax, length = 160)
    y = Float64[]
    log_s = Float64[]
    for θ in θs
        k, k1, k2 = cgf(θ)
        yθ = M * k1
        r = sqrt(max(2 * (θ * yθ - M * k), 0.0))
        u = θ * sqrt(M * k2)
        # 1 − Φ(r) + φ(r)(1/u − 1/r), with 1 − Φ(r) = erfc(r/√2)/2.
        tail = erfc(r / sqrt(2)) / 2 + exp(-r^2 / 2) / sqrt(2π) * (1 / u - 1 / r)
        tail > 0 || continue
        push!(y, yθ)
        push!(log_s, log(tail))
    end
    return y, log_s
end
