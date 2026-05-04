c = [-3; -2; -1; -5; 0; 0; 0]
A = [7 3 4 1 1 0 0 ;
     2 1 1 5 0 1 0 ;
     1 4 5 2 0 0 1 ]
b = [7; 3; 8]

include("book.jl")
include("simplex.jl")

println("=" ^ 60)
println("BOOK (enumeração de bases)")
println("=" ^ 60)
x_book, obj_book = BookMethod.simplex_method(c, A, b)
println("x* = ", x_book)
println("obj = ", obj_book)

println()
println("=" ^ 60)
println("SIMPLEX (partindo da origem)")
println("=" ^ 60)
x_simp, obj_simp = SimplexMethod.simplex_method(c, A, b)
println("x* = ", x_simp)
println("obj = ", obj_simp)
