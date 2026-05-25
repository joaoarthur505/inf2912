using AssignmentProblems
using JuMP
using HiGHS
using CSV
using DataFrames
using Dates
using Printf

const TIME_LIMIT = 10.0 

function solve_gap(data::AssignmentProblem, time_limit::Float64; sense=:Min)
    m = na(data)   # numero de agentes
    n = nj(data)   # numero de jobs

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, time_limit)

    @variable(model, x[1:m, 1:n], Bin)

    obj_expr = @expression(model, sum(data.costs[i, j] * x[i, j] for i in 1:m, j in 1:n))
    if sense == :Min
        @objective(model, Min, obj_expr)
    else
        @objective(model, Max, obj_expr)
    end
    
    # restrições 
    for i in 1:n
        @constraint(model, sum(x[j, i] for j in 1:m) == 1)
    end
    for j in 1:m
        @constraint(model, sum(x[j, i]*data.consumptions[j,i] for i in 1:n) <=  data.capacities[j])        
    end

    optimize!(model)

    st = termination_status(model)
    has_primal = primal_status(model) == MOI.FEASIBLE_POINT

    obj = has_primal ? objective_value(model) : Inf

    if st == MOI.OPTIMAL
        return :Optimal, obj
    elseif has_primal
        return :Feasible, obj
    elseif st == MOI.INFEASIBLE
        return :Infeasible, Inf
    else
        return :Other, Inf
    end
end

function relative_gap(obj, lit_lb, lit_ub, sense::Symbol)
    ref = sense == :Min ? lit_ub : lit_lb
    return (obj - ref) / abs(ref) * 100.0
end

const INSTANCES = [
    :d05100, :d05200,
    :d10100, :d10200,
    :a05100, :a05200,
    :a10100, :a10200,
    :a20100, :a20200,
    :b05100, :b05200,
    :b10100, :b10200,
    :b20100, :b20200,
    :c05100, :c05200,
    :c10100, :c10200,
    :e05100, :e05200,
    :e10100, :e10200,
]

results = DataFrame(
    instance   = String[],
    n_agents   = Int[],
    n_jobs     = Int[],
    sense      = String[],
    status     = String[],
    objective  = Float64[],
    lit_lb     = Float64[],
    lit_ub     = Float64[],
    gap_pct    = Float64[],
    time_s     = Float64[],
)

for inst in INSTANCES, sense in (:Min, :Max)
    data = loadAssignmentProblem(inst, sense)

    t0 = time()
    status, obj = solve_gap(data, TIME_LIMIT; sense=sense)
    elapsed = time() - t0

    gap = relative_gap(obj, data.lb, data.ub, sense)

    push!(results, (
        string(data.name),
        na(data),
        nj(data),
        string(sense),
        string(status),
        isfinite(obj) ? obj : NaN,
        Float64(data.lb == typemin(Int64) ? NaN : data.lb),
        Float64(data.ub == typemax(Int64) ? NaN : data.ub),
        gap,
        elapsed,
    ))

    @printf "%-10s  %-3s  status=%-10s  obj=%-10.1f  gap=%6.2f%%  t=%.1fs\n" data.name sense status obj gap elapsed
end

out_file = joinpath(@__DIR__, "resultados_gap.csv")
CSV.write(out_file, results)
println("\nResultados salvos em: $out_file")
println(results)
