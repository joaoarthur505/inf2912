include("simplex.jl")
using .SimplexMethod

# ---------------------------------------------------------------
# Problem 1 — mixed constraints: =, >= and <=
#
#   min  4 x1 + x2
#   s.t. 3 x1 +   x2  =  3
#        4 x1 + 3 x2 >=  6
#          x1 + 2 x2 <=  4
#        x1, x2 >= 0
#
# Optimum: x* = (2/5, 9/5),  z* = 17/5 = 3.4
# ---------------------------------------------------------------
c1      = [4.0, 1.0]
A1      = [3.0  1.0;
           4.0  3.0;
           1.0  2.0]
b1      = [3.0, 6.0, 4.0]
signs1  = [:eq, :ge, :le]

println("\n" * "="^60)
println("Problem 1 — mixed constraints")
println("="^60)
x1, z1 = simplex_method(c1, A1, b1, signs1)
println("\nx* = ", x1)
println("z* = ", z1)
println("(expected: x* ≈ [0.4, 1.8], z* ≈ 3.4)")

# ---------------------------------------------------------------
# Problem 2 — only <= constraints (same as TG1)
#
#   min -3 x1 - 2 x2 -  x3 - 5 x4
#   s.t. 7 x1 + 3 x2 + 4 x3 +   x4 <= 7
#        2 x1 +   x2 +   x3 + 5 x4 <= 3
#          x1 + 4 x2 + 5 x3 + 2 x4 <= 8
# ---------------------------------------------------------------
c2      = [-3.0, -2.0, -1.0, -5.0]
A2      = [7.0  3.0  4.0  1.0;
           2.0  1.0  1.0  5.0;
           1.0  4.0  5.0  2.0]
b2      = [7.0, 3.0, 8.0]
signs2  = [:le, :le, :le]

println("\n" * "="^60)
println("Problem 2 — only <= (same as TG1)")
println("="^60)
x2, z2 = simplex_method(c2, A2, b2, signs2)
println("\nx* = ", x2)
println("z* = ", z2)
