# Check whether the rotation kernel's inner @batch actually runs across
# multiple threads (or just on thread 1 because the outer per=core @batch
# has reserved all cores).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Random, Unitful, Polyester
import Unitful: Hz

println("nthreads: $(Threads.nthreads())  maxthreadid: $(Threads.maxthreadid())")
println()
println("Top-level @batch per=core across 1000 iterations:")
ids_top = Vector{Int}(undef, 1000)
@batch per=core for j in 1:1000
    ids_top[j] = Threads.threadid()
end
println("  unique thread ids seen: $(unique(ids_top))")

# Now simulate: outer @batch over a single PRN, inner @batch over cols.
# Use a function so the inner @batch is in its own scope.
function inner_kernel(out)
    @batch per=core for j in 1:length(out)
        out[j] = Threads.threadid()
    end
end

println()
println("Single PRN call → inner @batch:")
ids_inner = Vector{Int}(undef, 12000)
inner_kernel(ids_inner)
println("  unique inner thread ids: $(unique(ids_inner))")

println()
println("Outer @batch (1 iter) → calls inner kernel:")
ids_nested = Vector{Int}(undef, 12000)
@batch per=core for i in 1:1
    inner_kernel(ids_nested)
end
println("  unique inner thread ids: $(unique(ids_nested))")
