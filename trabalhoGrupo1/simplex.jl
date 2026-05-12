module SimplexMethod

  using LinearAlgebra, Printf

  export simplex_method

  mutable struct SimplexTableau
    z_c     ::Array{Float64} # z_j - c_j
    Y       ::Array{Float64} # inv(B) * A
    x_B     ::Array{Float64} # inv(B) * b
    obj     ::Float64        # c_B * x_B
    b_idx   ::Array{Int64}   # indices for basic variables x_B
  end

  function is_nonnegative(x::Vector)
    return length( x[ x .< 0] ) == 0
  end

  function is_nonpositive(z::Array)
    return length( z[ z .> 0] ) == 0
  end

  # Converts (A, b, signs) to standard form by adding slack/surplus/artificial variables.
  # signs: vector of :le (<=), :ge (>=), or :eq (=) for each constraint.
  # Returns the extended (A, b, initial basis, artificial column indices).
  function setup_problem(A, b, signs)
    m, n = size(A)
    A = Array{Float64}(copy(A))
    b = Array{Float64}(copy(b))
    signs = copy(signs)

    # Force b >= 0 by flipping rows where b[i] < 0 (and reversing the sign).
    for i in 1:m
      if b[i] < 0
        A[i, :] *= -1
        b[i]    *= -1
        signs[i] = signs[i] === :le ? :ge :
                   signs[i] === :ge ? :le : :eq
      end
    end

    n_slack = count(s -> s === :le || s === :ge, signs)
    n_art   = count(s -> s === :ge || s === :eq, signs)
    total   = n + n_slack + n_art

    A_ext = zeros(m, total)
    A_ext[:, 1:n] = A

    b_idx   = zeros(Int, m)
    art_idx = Int[]
    col_s   = n
    col_a   = n + n_slack

    for i in 1:m
      if signs[i] === :le            # slack +1
        col_s += 1
        A_ext[i, col_s] = 1.0
        b_idx[i] = col_s
      elseif signs[i] === :ge        # surplus -1 + artificial +1
        col_s += 1
        A_ext[i, col_s] = -1.0
        col_a += 1
        A_ext[i, col_a] = 1.0
        b_idx[i] = col_a
        push!(art_idx, col_a)
      else                           # :eq → artificial +1
        col_a += 1
        A_ext[i, col_a] = 1.0
        b_idx[i] = col_a
        push!(art_idx, col_a)
      end
    end

    return A_ext, b, b_idx, art_idx
  end

  function print_tableau(t::SimplexTableau)
    m, n = size(t.Y)

    hline0 = repeat("-", 6)
    hline1 = repeat("-", 7*n)
    hline2 = repeat("-", 7)
    hline = join([hline0, "+", hline1, "+", hline2])

    println(hline)

    @printf("%6s|", "")
    for j in 1:length(t.z_c)
      @printf("%6.2f ", t.z_c[j])
    end
    @printf("| %6.2f\n", t.obj)

    println(hline)

    for i in 1:m
      @printf("x[%2d] |", t.b_idx[i])
      for j in 1:n
        @printf("%6.2f ", t.Y[i,j])
      end
      @printf("| %6.2f\n", t.x_B[i])
    end

    println(hline)
  end

  function pivot_at!(t::SimplexTableau, entering::Int, exiting::Int)
    m, n = size(t.Y)

    # Pivoting: exiting-row, entering-column
    # updating exiting-row
    coef = t.Y[exiting, entering]
    t.Y[exiting, :] /= coef
    t.x_B[exiting] /= coef

    # updating other rows of Y
    for i in setdiff(1:m, exiting)
      coef = t.Y[i, entering]
      t.Y[i, :] -= coef * t.Y[exiting, :]
      t.x_B[i] -= coef * t.x_B[exiting]
    end

    # updating the row for the reduced costs
    coef = t.z_c[entering]
    t.z_c -= coef * t.Y[exiting, :]'
    t.obj -= coef * t.x_B[exiting]

    # Updating b_idx
    t.b_idx[ findfirst(t.b_idx .== t.b_idx[exiting]) ] = entering
  end

  function pivoting!(t::SimplexTableau)
    entering, exiting = pivot_point(t)
    println("Pivoting: entering = x_$entering, exiting = x_$(t.b_idx[exiting])")
    pivot_at!(t, entering, exiting)
  end

  function pivot_point(t::SimplexTableau)
    # Finding the entering variable index
    entering = findfirst( t.z_c .> 0)[2]
    if entering == 0
      error("Optimal")
    end

    # min ratio test / finding the exiting variable index
    pos_idx = findall( t.Y[:, entering] .> 0 )
    if length(pos_idx) == 0
      error("Unbounded")
    end
    exiting = pos_idx[ argmin( t.x_B[pos_idx] ./ t.Y[pos_idx, entering] ) ]

    return entering, exiting
  end

  function initialize(c, A, b, b_idx)
    c = Array{Float64}(c)
    A = Array{Float64}(A)
    b = Array{Float64}(b)

    m, n = size(A)

    B    = A[:, b_idx]
    Y    = inv(B) * A
    x_B  = inv(B) * b
    c_B  = c[b_idx]
    obj  = dot(c_B, x_B)

    # z_c is a row vector
    z_c = zeros(1, n)
    for j in 1:n
      z_c[1, j] = dot(c_B, Y[:, j]) - c[j]
    end

    return SimplexTableau(z_c, Y, x_B, obj, copy(b_idx))
  end

  function is_optimal(t::SimplexTableau)
    return is_nonpositive(t.z_c)
  end

  function simplex_method(c, A, b, signs)
    c_orig = Array{Float64}(c)

    A_ext, b_std, b_idx, art_idx = setup_problem(A, b, signs)
    n_orig = length(c_orig)
    total  = size(A_ext, 2)
    m      = size(A_ext, 1)

    # Phase 1: minimize sum of artificial variables
    println(repeat("=", 60))
    println("Phase 1")
    println(repeat("=", 60))

    c_p1 = zeros(total)
    c_p1[art_idx] .= 1.0

    c_ext = zeros(total)
    c_ext[1:n_orig] = c_orig

    tableau = initialize(c_p1, A_ext, b_std, b_idx)
    print_tableau(tableau)

    while !is_optimal(tableau)
      pivoting!(tableau)
      print_tableau(tableau)
    end

    if tableau.obj > 1e-6
      error("Infeasible: sum of artificials = $(tableau.obj) > 0")
    end

    # Drive out any artificial that stayed basic at value 0 (degeneracy).
    for i in 1:m
      if tableau.b_idx[i] in art_idx
        nonbasic_no_art = setdiff(setdiff(1:total, tableau.b_idx), art_idx)
        j = findfirst(j -> abs(tableau.Y[i, j]) > 1e-9, nonbasic_no_art)
        if j !== nothing
          println("Removing degenerate artificial: entering = x_$(nonbasic_no_art[j]), exiting = x_$(tableau.b_idx[i])")
          pivot_at!(tableau, nonbasic_no_art[j], i)
        end
      end
    end

    # Drop rows where an artificial is still basic (redundant constraints).
    rows_keep = [i for i in 1:m if !(tableau.b_idx[i] in art_idx)]

    # Phase 2: drop artificial columns and optimize the original objective.
    println(repeat("=", 60))
    println("Phase 2")
    println(repeat("=", 60))

    non_art  = setdiff(1:total, art_idx)
    col_map  = Dict(old => new for (new, old) in enumerate(non_art))
    A_p2     = A_ext[rows_keep, non_art]
    b_p2     = b_std[rows_keep]
    c_p2     = c_ext[non_art]
    b_idx_p2 = [col_map[tableau.b_idx[i]] for i in rows_keep]

    tableau2 = initialize(c_p2, A_p2, b_p2, b_idx_p2)
    print_tableau(tableau2)

    while !is_optimal(tableau2)
      pivoting!(tableau2)
      print_tableau(tableau2)
    end

    opt_x = zeros(n_orig)
    for i in 1:length(tableau2.b_idx)
      if tableau2.b_idx[i] <= n_orig
        opt_x[tableau2.b_idx[i]] = tableau2.x_B[i]
      end
    end

    return opt_x, tableau2.obj
  end

end
