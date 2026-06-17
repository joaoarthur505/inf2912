# Trabalho 4 - Geracao de Colunas (Dantzig-Wolfe) para o GAP

using AssignmentProblems
using JuMP
using HiGHS
using CSV
using DataFrames
using Printf

const TIME_LIMIT = 10.0
const CG_MAX_ITERS = 1000
const EPS = 1e-6

# mochila 0-1 por DP: maximiza sum profit[j]*z[j] s.a. sum weight[j]*z[j] <= cap
# retorna (itens escolhidos, lucro)
function knapsack(profit::Vector{Float64}, weight::Vector{Int64}, cap::Int64)
    items = [j for j in eachindex(profit) if profit[j] > EPS && weight[j] <= cap]
    if isempty(items)
        return Int64[], 0.0
    end

    k = length(items)
    dp = zeros(Float64, k + 1, cap + 1)
    for i in 1:k
        wj = weight[items[i]]
        pj = profit[items[i]]
        for c in 0:cap
            best = dp[i, c+1]
            if wj <= c
                alt = dp[i, c-wj+1] + pj
                if alt > best
                    best = alt
                end
            end
            dp[i+1, c+1] = best
        end
    end

    chosen = Int64[]
    c = cap
    for i in k:-1:1
        if dp[i+1, c+1] > dp[i, c+1] + 1e-9
            push!(chosen, items[i])
            c -= weight[items[i]]
        end
    end
    return chosen, dp[k+1, cap+1]
end

function read_solution(model)
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

# solucao inicial gulosa: cada tarefa no agente viavel mais barato (1 padrao por agente)
function initial_patterns(data::AssignmentProblem, c::Matrix{Float64})
    m = na(data)
    n = nj(data)
    remaining = copy(data.capacities)
    patterns = [Int64[] for _ in 1:m]
    for j in 1:n
        best_i = 0
        best_c = Inf
        for i in 1:m
            if data.consumptions[i, j] <= remaining[i] && c[i, j] < best_c
                best_c = c[i, j]
                best_i = i
            end
        end
        if best_i > 0
            push!(patterns[best_i], j)
            remaining[best_i] -= data.consumptions[best_i, j]
        end
    end
    return patterns
end

function solve_colgen(data::AssignmentProblem, sense::Symbol, time_limit::Float64)
    t0 = time()
    m = na(data)
    n = nj(data)

    sgn = sense == :Min ? 1.0 : -1.0
    c = sgn .* Float64.(data.costs)        # custo interno: sempre minimizamos
    w = data.consumptions
    cap = data.capacities
    bigM = sum(abs, c) + 1.0

    master = Model(HiGHS.Optimizer)
    set_silent(master)

    @variable(master, art[1:n] >= 0)               # artificiais (cobertura)
    @variable(master, base[1:m] >= 0)              # coluna inicial (gulosa) de cada agente
    @constraint(master, cover[j in 1:n], art[j] == 1)
    @constraint(master, conv[i in 1:m], base[i] == 1)
    @objective(master, Min, bigM * sum(art))

    cols = Any[]
    function register_column!(i, S, v)
        if !isempty(S)
            set_objective_coefficient(master, v, sum(c[i, j] for j in S))
            for j in S
                set_normalized_coefficient(cover[j], v, 1.0)
            end
        end
        push!(cols, (i = i, S = S, var = v))
    end

    function add_column!(i, S)
        v = @variable(master, lower_bound = 0.0)
        set_normalized_coefficient(conv[i], v, 1.0)
        register_column!(i, S, v)
    end

    # warm start: padroes gulosos como colunas-base
    for (i, S) in enumerate(initial_patterns(data, c))
        register_column!(i, S, base[i])
    end

    n_iters = 0
    lp_bound = NaN
    seen = Set{Tuple{Int64,Vector{Int64}}}()
    for it in 1:CG_MAX_ITERS
        optimize!(master)
        if termination_status(master) != MOI.OPTIMAL
            break
        end
        n_iters = it
        lp_bound = sgn * objective_value(master)   # bound de Dantzig-Wolfe

        π = dual.(cover)
        μ = dual.(conv)

        found = false
        for i in 1:m
            profit = [π[j] - c[i, j] for j in 1:n]
            S, P = knapsack(profit, w[i, :], cap[i])
            if isempty(S) || -P - μ[i] >= -EPS     # custo reduzido nao-negativo
                continue
            end
            key = (i, sort(S))
            if key in seen                          # ja gerada antes
                continue
            end
            push!(seen, key)
            add_column!(i, S)
            found = true
        end

        if !found || time() - t0 > time_limit
            break
        end
    end

    # fase inteira: mestre restrito com lambda binario, sem artificiais
    for col in cols
        set_binary(col.var)
    end
    for j in 1:n
        set_upper_bound(art[j], 0.0)
    end
    set_time_limit_sec(master, max(time_limit - (time() - t0), 0.1))
    optimize!(master)
    elapsed = time() - t0

    status, obj_internal = read_solution(master)
    return (status = status, obj = sgn * obj_internal, time = elapsed,
            lp_bound = lp_bound, n_cols = length(cols), n_iters = n_iters)
end

const INSTANCES = [
    :a05100, :a05200, :a10100, :a10200, :a20100, :a20200,
    :b05100, :b05200, :b10100, :b10200, :b20100, :b20200,
    :c05100, :c05200, :c10100, :c10200,
    :d05100, :d05200, :d10100, :d10200,
    :e05100, :e05200, :e10100, :e10200,
]

function run_all(insts = INSTANCES; senses = (:Min, :Max), save = true)
    results = DataFrame(
        instance  = String[],
        sense     = String[],
        status    = String[],
        objective = Float64[],
        lp_bound  = Float64[],
        n_cols    = Int[],
        gap_pct   = Float64[],
        time_s    = Float64[],
    )

    @printf "%-9s %-3s | %-9s | %-9s | %-6s | %-8s | %s\n" "inst" "sns" "obj" "bound" "cols" "gap%" "tempo"
    println(repeat("-", 78))

    for inst in insts, sense in senses
        data = loadAssignmentProblem(inst, sense)
        if data === nothing
            continue
        end

        cg = solve_colgen(data, sense, TIME_LIMIT)
        gap = relative_gap(cg.obj, data.lb, data.ub, sense)

        push!(results, (
            string(data.name), string(sense),
            string(cg.status), isfinite(cg.obj) ? cg.obj : NaN,
            cg.lp_bound, cg.n_cols, gap, cg.time,
        ))

        @printf "%-9s %-3s | %-9.1f | %-9.1f | %-6d | %7.3f%% | %.2fs\n" data.name sense cg.obj cg.lp_bound cg.n_cols gap cg.time
    end

    if save
        out_file = joinpath(@__DIR__, "resultados_colgen.csv")
        CSV.write(out_file, results)
        println("\nResultados salvos em: ", out_file)
    end

    n = nrow(results)
    if n > 0
        cg_opt = count(==("Optimal"), results.status)
        @printf "\nResumo (%d execucoes): otimos=%d, media de colunas=%.0f, tempo total=%.1fs\n" n cg_opt (sum(results.n_cols)/n) sum(results.time_s)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
