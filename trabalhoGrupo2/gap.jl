using AssignmentProblems
using JuMP
using HiGHS       # troca pelo solver que vocês usarem (CPLEX, Gurobi, etc.)
using CSV
using DataFrames
using Dates

const TIME_LIMIT = 300.0  # segundos por instância

# ──────────────────────────────────────────────────────────────
# TODO: preencha aqui o modelo MIP de vocês
# Recebe um AssignmentProblem, retorna (status, obj_value, bound)
#   status  → :Optimal | :Feasible | :Infeasible | :Other
#   obj     → melhor valor inteiro encontrado (Inf se nenhum)
#   bound   → melhor bound dual do solver (para calcular gap interno)
# ──────────────────────────────────────────────────────────────
function solve_gap(data::AssignmentProblem, time_limit::Float64)
    m = na(data)   # número de agentes
    n = nj(data)   # número de jobs

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, time_limit)

    # --- variáveis ---
    @variable(model, x[1:m, 1:n], Bin)

    # --- objetivo (minimização) ---
    @objective(model, Min, sum(data.costs[i, j] * x[i, j] for i in 1:m, j in 1:n))

    # --- restrições ---
    # TODO: adicione aqui as restrições do GAP

    optimize!(model)

    st = termination_status(model)
    has_primal = primal_status(model) == MOI.FEASIBLE_POINT

    obj   = has_primal ? objective_value(model)   : Inf
    bound = has_dual_status(model) ? dual_objective_value(model) : -Inf

    if st == MOI.OPTIMAL
        return :Optimal, obj, bound
    elseif has_primal
        return :Feasible, obj, bound
    elseif st == MOI.INFEASIBLE
        return :Infeasible, Inf, Inf
    else
        return :Other, Inf, Inf
    end
end

# ──────────────────────────────────────────────────────────────
# Gap relativo (%) usando o bound dual do solver (mais preciso)
# Fallback para lb da literatura se o solver não tiver bound.
# ──────────────────────────────────────────────────────────────
function relative_gap(obj, solver_bound, lit_lb)
    # Usa bound do próprio solver quando disponível
    lb = isfinite(solver_bound) ? solver_bound : lit_lb
    if !isfinite(obj) || !isfinite(lb) || lb == 0
        return NaN
    end
    return (obj - lb) / abs(lb) * 100.0
end

# ──────────────────────────────────────────────────────────────
# Instâncias que queremos rodar — edite à vontade
# ──────────────────────────────────────────────────────────────
const INSTANCES = [
    :a05100, :a05200,
    :a10100, :a10200,
    :a20100, :a20200,
    :b05100, :b05200,
    :b10100, :b10200,
    :b20100, :b20200,
    :c05100, :c05200,
    :c10100, :c10200,
    :d05100, :d05200,
    :d10100, :d10200,
    :e05100, :e05200,
    :e10100, :e10200,
]

# ──────────────────────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────────────────────
results = DataFrame(
    instance   = String[],
    n_agents   = Int[],
    n_jobs     = Int[],
    status     = String[],
    objective  = Float64[],
    lit_lb     = Float64[],
    lit_ub     = Float64[],
    gap_pct    = Float64[],   # gap relativo (%)
    time_s     = Float64[],
)

for inst in INSTANCES
    data = loadAssignmentProblem(inst)
    if data === nothing
        @warn "Instância $inst não encontrada, pulando."
        continue
    end

    println("Resolvendo $(data.name)  ($(na(data)) agentes × $(nj(data)) jobs)...")

    t0 = time()
    status, obj, bound = solve_gap(data, TIME_LIMIT)
    elapsed = time() - t0

    gap = relative_gap(obj, bound, Float64(data.lb))

    push!(results, (
        string(data.name),
        na(data),
        nj(data),
        string(status),
        isfinite(obj) ? obj : NaN,
        Float64(data.lb == typemin(Int64) ? NaN : data.lb),
        Float64(data.ub == typemax(Int64) ? NaN : data.ub),
        gap,
        elapsed,
    ))

    @printf "  → status=%-10s  obj=%-10.1f  gap=%6.2f%%  t=%.1fs\n" status obj gap elapsed
end

# ──────────────────────────────────────────────────────────────
# Exporta CSV
# ──────────────────────────────────────────────────────────────
out_file = joinpath(@__DIR__, "resultados_gap.csv")
CSV.write(out_file, results)
println("\nResultados salvos em: $out_file")
println(results)
