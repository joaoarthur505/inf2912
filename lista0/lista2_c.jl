using JuMP
using HiGHS

P = [1,2]
R = [1,2]

v = [4, 1]
d = [16, 12]
c = [
    8 2
    5 2
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