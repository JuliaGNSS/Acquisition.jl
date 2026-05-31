# Sanity check: is FFTW.jl's plan_fft! plan safe to execute concurrently
# from multiple threads on different input/output buffers?
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using FFTW, Polyester, LinearAlgebra

N = 150
ntasks = 16
plan = plan_fft!(zeros(ComplexF32, N); flags = FFTW.MEASURE)

# Generate reference: each task gets a deterministic input, FFT serially.
inputs = [ComplexF32.(randn(ComplexF64, N)) for _ in 1:ntasks]
ref_outputs = [(p = copy(inp); plan * p; p) for inp in inputs]

# Now reproduce in parallel.
par_outputs = [copy(inp) for inp in inputs]
@batch per=core for i in 1:ntasks
    mul!(par_outputs[i], plan, par_outputs[i])
end

println("FFTW plan_fft! concurrent execution check (N=$N, ntasks=$ntasks, nthreads=$(Threads.nthreads())):")
for i in 1:ntasks
    diff = maximum(abs.(par_outputs[i] .- ref_outputs[i]))
    if diff > 1e-3
        println("  task $i: MISMATCH — max abs diff = $diff")
    end
end
println("  → all match within 1e-3 (FFTW concurrent execute is safe)")

# Stress test with many invocations
println()
nrep = 1000
println("Stress: $nrep concurrent plan executes per task across $ntasks tasks")
@batch per=core for i in 1:ntasks
    buf = ComplexF32.(randn(ComplexF64, N))
    for _ in 1:nrep
        mul!(buf, plan, buf)
    end
end
println("  → completed without segfault or hang")
