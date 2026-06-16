# Trabalho 3 - Planos de Cortes (desigualdades de cobertura) para o GAP

using AssignmentProblems
using JuMP
using HiGHS
using CSV
using DataFrames
using Printf

const TIME_LIMIT = 10.0
const CP_MAX_ITERS = 100
const EPS = 1e-6

# separa uma cobertura violada da mochila sum_j w[j]*x[j] <= cap
function separate_cover(w::Vector{Int64}, cap::Int64, xstar::Vector{Float64})
    n = length(w)

    cand = [j for j in 1:n if w[j] > 0]
    if isempty(cand)
        return nothing
    end

    sort!(cand, by = j -> (1.0 - xstar[j]) / w[j])

    C = Int64[]
    weight = 0
    for j in cand
        push!(C, j)
        weight += w[j]
        if weight > cap
            break
        end
    end

    if weight <= cap
        return nothing
    end

    # torna a cobertura minimal removendo o item de menor xstar
    improved = true
    while improved
        improved = false
        best_idx = 0
        best_x = Inf
        for (idx, j) in enumerate(C)
            if weight - w[j] > cap
                if xstar[j] < best_x
                    best_x = xstar[j]
                    best_idx = idx
                end
            end
        end
        if best_idx > 0
            weight -= w[C[best_idx]]
            deleteat!(C, best_idx)
            improved = true
        end
    end

    return sum(xstar[j] for j in C) > length(C) - 1 + EPS ? C : nothing
end

function build_model(data::AssignmentProblem; sense::Symbol, is_relaxed::Bool)
    m = na(data)
    n = nj(data)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    if is_relaxed
        @variable(model, 0 <= x[1:m, 1:n] <= 1)
    else
        @variable(model, x[1:m, 1:n], Bin)
    end

    obj = @expression(model, sum(data.costs[i, j] * x[i, j] for i in 1:m, j in 1:n))
    if sense == :Min
        @objective(model, Min, obj)
    else
        @objective(model, Max, obj)
    end

    @constraint(model, [j in 1:n], sum(x[i, j] for i in 1:m) == 1)
    @constraint(model, [i in 1:m], sum(data.consumptions[i, j] * x[i, j] for j in 1:n) <= data.capacities[i])

    return model, x
end

function cutting_plane_loop!(model, x, data::AssignmentProblem)
    m = na(data)
    n = nj(data)

    lp_root = NaN
    lp_cuts = NaN
    n_cuts = 0
    n_iters = 0

    for it in 1:CP_MAX_ITERS
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            break
        end

        n_iters = it
        z = objective_value(model)
        if it == 1
            lp_root = z
        end
        lp_cuts = z

        xval = value.(x)
        found = false
        for i in 1:m
            w = data.consumptions[i, :]
            xstar = [xval[i, j] for j in 1:n]
            C = separate_cover(w, data.capacities[i], xstar)
            if C === nothing
                continue
            end

            @constraint(model, sum(x[i, j] for j in C) <= length(C) - 1)
            n_cuts += 1
            found = true
        end

        if !found
            break
        end
    end

    return lp_root, lp_cuts, n_cuts, n_iters
end

function solve_cutting_planes(data::AssignmentProblem, sense::Symbol, time_limit::Float64)
    t0 = time()
    model, x = build_model(data; sense = sense, is_relaxed = true)

    lp_root, lp_cuts, n_cuts, n_iters = cutting_plane_loop!(model, x, data)

    # resolve o inteiro mantendo os cortes; desconta o tempo ja gasto
    set_binary.(x)
    set_time_limit_sec(model, max(time_limit - (time() - t0), 0.1))
    optimize!(model)
    elapsed = time() - t0

    status, obj = read_solution(model)
    return (status = status, obj = obj, time = elapsed,
            lp_root = lp_root, lp_cuts = lp_cuts, n_cuts = n_cuts, n_iters = n_iters)
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
        lp_root   = Float64[],
        lp_cuts   = Float64[],
        n_cuts    = Int[],
        gap_pct   = Float64[],
        time_s    = Float64[],
    )

    @printf "%-9s %-3s | %-9s | root -> cortes   (cortes/iter) | %-8s | %s\n" "inst" "sns" "obj" "gap%" "tempo"
    println(repeat("-", 82))

    for inst in insts, sense in senses
        data = loadAssignmentProblem(inst, sense)
        if data === nothing
            continue
        end

        cp  = solve_cutting_planes(data, sense, TIME_LIMIT)
        gap = relative_gap(cp.obj, data.lb, data.ub, sense)

        push!(results, (
            string(data.name), string(sense),
            string(cp.status), isfinite(cp.obj) ? cp.obj : NaN,
            cp.lp_root, cp.lp_cuts, cp.n_cuts,
            gap, cp.time,
        ))

        @printf "%-9s %-3s | %-9.1f | %8.2f -> %-8.2f (%3d/%-2d) | %7.3f%% | %.2fs\n" data.name sense cp.obj cp.lp_root cp.lp_cuts cp.n_cuts cp.n_iters gap cp.time
    end

    if save
        out_file = joinpath(@__DIR__, "resultados_cortes.csv")
        CSV.write(out_file, results)
        println("\nResultados salvos em: ", out_file)
    end

    n = nrow(results)
    if n > 0
        avg_cuts = sum(results.n_cuts) / n
        cp_opt = count(==("Optimal"), results.status)
        @printf "\nResumo (%d execucoes): otimos=%d, media de cortes=%.1f, tempo total=%.1fs\n" n cp_opt avg_cuts sum(results.time_s)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
