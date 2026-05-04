using JuMP
using HiGHS

P = [1,2]

v = [-3, -2]
d = [-3, -2, -6, 3]
c = [
    -1 0
    0 -1
    -1 -1
    1 -1 
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
    println("z = ", (-1)*objective_value(model))
    for j in P
        println("x[$j] = ", value(x[j]))
    end
end