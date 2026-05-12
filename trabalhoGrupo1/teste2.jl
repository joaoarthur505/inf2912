include("simplex.jl")
using .SimplexMethod

c1      = [2.0, 5.0]
A1      = [3.0  1.0;
           6.0  4.0]
b1      = [3.0, 7.0]
signs1  = [:eq, :le]

println("\n" * "="^60)
println("Problem 1 — mixed constraints")
println("="^60)
x1, z1 = simplex_method(c1, A1, b1, signs1)
println("\nx* = ", x1)
println("z* = ", z1)

