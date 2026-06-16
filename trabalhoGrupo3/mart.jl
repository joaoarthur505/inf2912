using TSPLIB

using JuMP
using Gurobi

using Graphs
using GraphPlot
import Cairo, Fontconfig

using CVRPSEP

const Variables = Containers.DenseAxisArray{VariableRef, 2}
const Solution = Vector{Tuple{Int64, Int64, Float64}}
const GRB_ENV = Gurobi.Env()
const EPS = 1e-4

cut_manager = CutManager()

function buildModel(data::TSP; is_relaxed::Bool = false, is_silent::Bool = true)
    n = data.dimension
    V = 1:n
    c = data.weights

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    if is_relaxed
        @variable(model, 0 <= x[V, V] <= 1)
    else
        @variable(model, x[V, V], Bin)
    end

    @objective(model, Min, sum(c[i, j]x[i, j] for i in V, j in V))

    @constraint(model, [j in V], sum(x[i, j] for i in V) == 1)
    @constraint(model, [i in V], sum(x[i, j] for j in V) == 1)
    @constraint(model, [i in V], x[i, i] == 0)

    # println(model)
    if is_silent set_silent(model) end
    set_time_limit_sec(model, 60)

    # CVRPSEP is not thread safe
    set_optimizer_attribute(model, "Threads", 1)

    return model, x
end

function buildSolution(data::TSP, sol_x::Containers.DenseAxisArray{Float64, 2})::Solution
    solution = Solution()
    for i in 1:data.dimension
        for j in 1:data.dimension
            if i < j && (sol_x[i, j] > EPS || sol_x[j, i] > EPS)
                push!(solution, (i, j, sol_x[i, j] + sol_x[j, i]))
            end
        end
    end
    return solution
end

function getViolation(data::TSP, solution::Solution, S::Vector{Int64})::Float64
    inS = falses(data.dimension)
    inS[S] .= true

    lhs = 0.0
    rhs = length(S) - 1
    for (from, to, value) in solution
        if inS[from] && inS[to]
            lhs += value
        end
    end
    return lhs - rhs
end

function separateFractional(data::TSP, solution::Solution)::Vector{Vector{Int64}}
    n = data.dimension

    demand = ones(Int64, n)
    demand[1] = 0
    capacity = n - 1

    edge_tail = Int64[]
    edge_head = Int64[]
    edge_x = Float64[]

    for (from, to, value) in solution
        push!(edge_tail, from)
        push!(edge_head, to)
        push!(edge_x, value)
    end

    temp, _ = rounded_capacity_inequalities!(cut_manager, demand, capacity, edge_tail, edge_head, edge_x; max_n_cuts = 10, integrality_tolerance = 1e-4)

    Ss = Vector{Int64}[]
    for S in temp
        if getViolation(data, solution, S) < EPS continue end

        if length(S) <= n ÷ 2
            push!(Ss, S)
        else
            Sset = Set(S)
            push!(Ss, [ v for v in 1:n if !(v in Sset) ])
        end
    end

    return Ss
end

function separateInteger(data::TSP, solution::Solution)::Vector{Vector{Int64}}
    g = buildGraph(data, solution)
    # plotGraph(data, g)

    temp = connected_components(g)

    Ss = Vector{Int64}[]
    for S in temp
        if getViolation(data, solution, S) < EPS continue end

        if 1 < length(S) <= data.dimension ÷ 2
            push!(Ss, S)
        end
    end

    return Ss
end

function insertConstraints(model::Model, x::Variables, Ss::Vector{Vector{Int64}})::Bool
    found = false
    for S in Ss
        @constraint(model, sum(x[i, j] for i in S, j in S) <= length(S) - 1)
        found = true
    end
    return found
end

function callback(cb_data, cb_where::Cint, data::TSP, model::Model, x::Variables)
    # You can select where the callback is run
    if cb_where != GRB_CB_MIPSOL && cb_where != GRB_CB_MIPNODE
        return
    end

    if cb_where == GRB_CB_MIPSOL
        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        sol_x = callback_value.(cb_data, x)
        solution = buildSolution(data, sol_x)

        Ss = separateInteger(data, solution)
        for S in Ss
            con = @build_constraint(sum(x[i, j] for i in S, j in S) <= length(S) - 1)
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    elseif cb_where == GRB_CB_MIPNODE
        node_status = Ref{Cint}()
        ret = Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_STATUS, node_status)
        if ret != 0 || node_status[] != Gurobi.GRB_OPTIMAL
            return
        end

        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        sol_x = callback_value.(cb_data, x)
        solution = buildSolution(data, sol_x)

        Ss = separateFractional(data, solution)
        for S in Ss
            con = @build_constraint(sum(x[i, j] for i in S, j in S) <= length(S) - 1)
            MOI.submit(model, MOI.UserCut(cb_data), con)
        end
    end
end

function buildGraph(data::TSP, solution::Solution)::SimpleGraph
    g = SimpleGraph(data.dimension)
    for (from, to, _) in solution
        add_edge!(g, from, to)
    end
    return g
end

function plotGraph(data::TSP, g::SimpleGraph)
    coordx = data.nodes[:, 1]
    coordy = data.nodes[:, 2]
    display(gplot(g, coordx, coordy, nodelabel=1:data.dimension, font_family="Consolas"))
end

function solveOnlyInteger(data::TSP)
    model, x = buildModel(data)

    found = true
    while found
        optimize!(model)
        println("z = ", objective_value(model))

        sol_x = value.(x)
        solution = buildSolution(data, sol_x)

        Ss = separateInteger(data, solution)
        found = insertConstraints(model, x, Ss)
    end
end

function solveFractionalAndInteger(data::TSP)
    fractional = true
    model, x = buildModel(data; is_relaxed = true)

    found = true
    while found
        optimize!(model)
        println("z = ", objective_value(model))

        sol_x = value.(x)
        solution = buildSolution(data, sol_x)

        if fractional
            Ss = separateFractional(data, solution)
            found = insertConstraints(model, x, Ss)
            if !found
                println("Removing relaxation...")
                fractional = false
                set_binary.(x)
                found = true
            end
        else
            Ss = separateInteger(data, solution)
            found = insertConstraints(model, x, Ss)
        end
    end
end

function solveCallback(data::TSP)
    model, x = buildModel(data)

    function innerCallback(cb_data, cb_where::Cint)
        callback(cb_data, cb_where, data, model, x)
    end

    MOI.set(model, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
    MOI.set(model, Gurobi.CallbackFunction(), innerCallback)
    optimize!(model)

    println("z = ", objective_value(model))
end

function solveFractionalAndCallback(data::TSP)
    model, x = buildModel(data; is_relaxed = true)

    found = true
    while found
        optimize!(model)
        println("z = ", objective_value(model))

        sol_x = value.(x)
        solution = buildSolution(data, sol_x)

        Ss = separateFractional(data, solution)
        found = insertConstraints(model, x, Ss)
        if !found
            println("Removing relaxation...")
            set_binary.(x)

            function innerCallback(cb_data, cb_where::Cint)
                callback(cb_data, cb_where, data, model, x)
            end

            MOI.set(model, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
            MOI.set(model, Gurobi.CallbackFunction(), innerCallback)

            optimize!(model)

            println("z = ", objective_value(model))
        end
    end
end

function main()
    data = readTSPLIB(:a280)
    println("Read ", data.name, " - opt = ", data.optimal)

    println(" >>> Only Integer <<<")
    @time solveOnlyInteger(data)

    println(" >>> Fractional and Integer <<<")
    @time solveFractionalAndInteger(data)

    println(" >>> Callback <<<")
    @time solveCallback(data)

    println(" >>> Fractional and Callback <<<")
    @time solveFractionalAndCallback(data)
end

main()