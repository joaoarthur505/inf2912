# =============================================================================
#  Trabalho 3 - Metodo de Planos de Cortes para o GAP (Generalized Assignment
#  Problem), usando desigualdades de COBERTURA (cover inequalities).
#
#  GAP:
#     min/max  sum_{i,j} c[i,j] x[i,j]
#     s.a.     sum_i x[i,j] = 1                         (cada tarefa j a 1 agente)
#              sum_j w[i,j] x[i,j] <= cap[i]            (capacidade do agente i)
#              x[i,j] in {0,1}
#
#  Cada restricao de capacidade e' uma MOCHILA. Para uma cobertura C (subconjunto
#  de tarefas) tal que   sum_{j in C} w[i,j] > cap[i],  vale a desigualdade
#  valida (corte):       sum_{j in C} x[i,j] <= |C| - 1.
#
#  Metodo (analogo ao gap.jl/mart.jl do trabalho anterior, mas para o GAP):
#     1. Resolve a relaxacao linear (0 <= x <= 1).
#     2. Para cada agente, SEPARA coberturas violadas pela solucao fracionaria.
#     3. Adiciona os cortes e re-resolve. Repete ate nao haver corte violado.
#     4. Com a formulacao fortalecida, resolve o problema inteiro (branch & bound).
#
#  Saida (resultados_cortes.csv): valor da solucao, status, tempo, gap e ainda
#  o bound da relaxacao na raiz (lp_root) vs. apos os cortes (lp_cuts) e o
#  numero de cortes/iteracoes. As colunas status/objective/time_s/gap_pct usam
#  os mesmos nomes do trabalho anterior (trabalhoGrupo2/resultados_gap.csv),
#  para comparar os dois CSVs diretamente.
# =============================================================================

using AssignmentProblems
using JuMP
using HiGHS
using CSV
using DataFrames
using Printf

const TIME_LIMIT = 10.0     # limite de tempo (s) para a fase inteira (igual ao trab. anterior)
const CP_MAX_ITERS = 100    # maximo de iteracoes do laco de planos de cortes
const EPS = 1e-6            # tolerancia numerica

# -----------------------------------------------------------------------------
#  Separacao de coberturas para UMA restricao de mochila.
#
#  Mochila:  sum_j w[j] x[j] <= cap.  Solucao fracionaria xstar.
#  Procura uma cobertura C (sum_{j in C} w[j] > cap) com violacao
#       sum_{j in C} xstar[j] - (|C| - 1) > 0   <=>   sum_{j in C}(1 - xstar[j]) < 1.
#
#  Heuristica gulosa classica:
#     - monta uma cobertura adicionando itens com menor "folga por peso"
#       (1 - xstar[j]) / w[j]  (itens que o LP "usou", xstar ~ 1, sao baratos);
#     - torna a cobertura MINIMAL removendo itens de maior folga (1 - xstar[j])
#       enquanto continuar sendo cobertura.
#  Retorna o vetor de indices da cobertura, ou `nothing` se nao ha violacao.
# -----------------------------------------------------------------------------
function separate_cover(w::Vector{Int64}, cap::Int64, xstar::Vector{Float64})
    n = length(w)

    # candidatos: itens com peso > 0 (peso 0 nunca ajuda a estourar a capacidade)
    cand = [j for j in 1:n if w[j] > 0]
    if isempty(cand)
        return nothing
    end

    # ordena por folga por unidade de peso (crescente)
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

    if weight <= cap     # nem todos os itens estouram a capacidade
        return nothing
    end

    # torna minimal: remove o item de menor xstar que ainda mantem cobertura
    # (menor xstar == maior folga 1-xstar == maior ganho de violacao ao remover)
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

    # viola se sum_{j in C} xstar[j] > |C| - 1
    return sum(xstar[j] for j in C) > length(C) - 1 + EPS ? C : nothing
end

# -----------------------------------------------------------------------------
#  Constroi o modelo do GAP. is_relaxed=true => 0 <= x <= 1 (relaxacao linear).
# -----------------------------------------------------------------------------
function build_model(data::AssignmentProblem; sense::Symbol, is_relaxed::Bool)
    m = na(data)        # agentes
    n = nj(data)        # tarefas

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

# -----------------------------------------------------------------------------
#  Laco de planos de cortes sobre a relaxacao linear.
#  Adiciona coberturas violadas (1 por agente por iteracao) ate convergir.
#  Retorna (lp_root, lp_cuts, n_cuts, n_iters).
# -----------------------------------------------------------------------------
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
            (lp_root = z)
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

        if !found      # nenhum corte violado => convergiu
            break
        end
    end

    return lp_root, lp_cuts, n_cuts, n_iters
end

# -----------------------------------------------------------------------------
#  Resolve o GAP por Planos de Cortes:
#     relaxacao -> cortes de cobertura -> resolve inteiro com a formulacao forte.
# -----------------------------------------------------------------------------
function solve_cutting_planes(data::AssignmentProblem, sense::Symbol, time_limit::Float64)
    t0 = time()
    model, x = build_model(data; sense = sense, is_relaxed = true)

    lp_root, lp_cuts, n_cuts, n_iters = cutting_plane_loop!(model, x, data)

    # fase inteira: mantem os cortes, exige x binario.
    # desconta o tempo gasto no laco de cortes para o orcamento total ser justo vs MIP puro.
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

# gap relativo (%) ao melhor bound da literatura (ub p/ Min, lb p/ Max)
function relative_gap(obj, lit_lb, lit_ub, sense::Symbol)
    ref = sense == :Min ? lit_ub : lit_lb
    return (obj - ref) / abs(ref) * 100.0
end

# =============================================================================
#  Execucao: roda o metodo de Planos de Cortes e salva os resultados.
#  A comparacao com o trabalho anterior e' feita entre os CSVs
#  (trabalhoGrupo2/resultados_gap.csv  vs  resultados_cortes.csv): as colunas
#  status / objective / time_s / gap_pct tem os mesmos nomes nos dois.
# =============================================================================

# instancias avaliadas (mesmas do trabalho anterior). Cada uma roda em :Min e :Max.
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
        lp_root   = Float64[],     # bound da relaxacao na raiz (antes dos cortes)
        lp_cuts   = Float64[],     # bound apos os cortes de cobertura
        n_cuts    = Int[],         # numero de cortes de cobertura adicionados
        gap_pct   = Float64[],     # gap (%) vs melhor bound da literatura
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

# roda automaticamente quando executado como script (julia gap_cortes.jl)
if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
