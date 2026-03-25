using JuMP
using HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, xm >= 0)
@variable(model, xc >= 0)

@objective(model, Max, 1000xm + 500xc)

@constraint(model, 3xm + 5xc <= 40)
@constraint(model, 7xm + 4xc <= 60)

println(model)
set_silent(model)
optimize!(model)

println("z = ", objective_value(model))
println("xm = ", value(xm))
println("xc = ", value(xc))