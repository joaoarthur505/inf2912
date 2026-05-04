using JuMP
using HiGHS

P = [2, 1]

v = [2, 1]
d = [10, -60, 18, 44]
c = [
    0 1
    -2 -5
    1 1
    3 1
]

model = Model(HiGHS.Optimizer) 

@variable(model, x[P] >= 0)

# @objective(model, Max, sum(v[j]x[j] for j in P))
@objective(model, Max, v'x)


# @constraint(model, [i in R], sum(c[i, j]x[j] for j in P) <= d[i])
@constraint(model, c * x .<= d)

println(model)
set_silent(model)
optimize!(model)

println(termination_status(model))

if termination_status(model) == OPTIMAL
    println("z = ", objective_value(model))
    for j in P
        println("x[$j] = ", value(x[j]))
    end
end