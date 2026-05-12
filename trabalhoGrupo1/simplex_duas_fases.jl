module SimplexDuasFases

  using LinearAlgebra, Printf

  export simplex_duas_fases

  const TOL = 1e-9

  mutable struct SimplexTableau
    z_c   ::Array{Float64}   # z_j - c_j (linha)
    Y     ::Array{Float64}   # inv(B) * A
    x_B   ::Array{Float64}   # inv(B) * b
    obj   ::Float64          # c_B * x_B
    b_idx ::Array{Int64}     # índices das variáveis básicas
  end

  is_nonpositive(z) = all(z .<= TOL)

  function print_tableau(t::SimplexTableau, titulo::String="")
    m, n = size(t.Y)
    if !isempty(titulo)
      println(titulo)
    end
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

  # Constrói o tableau a partir de uma base qualquer (não exige identidade nas últimas colunas).
  function build_tableau(c, A, b, b_idx)
    m, n = size(A)
    B = A[:, b_idx]
    Binv = inv(B)
    Y = Binv * A
    x_B = Binv * b
    c_B = c[b_idx]
    obj = dot(c_B, x_B)
    z_c = zeros(1, n)
    for j in 1:n
      z_c[1, j] = dot(c_B, Y[:, j]) - c[j]
    end
    return SimplexTableau(z_c, Y, x_B, obj, copy(b_idx))
  end

  function is_optimal(t::SimplexTableau)
    return is_nonpositive(t.z_c)
  end

  function pivot_point(t::SimplexTableau)
    m, n = size(t.Y)
    entering = 0
    for j in 1:n
      if t.z_c[j] > TOL
        entering = j
        break
      end
    end
    if entering == 0
      error("Otimo")
    end

    pos_idx = findall(t.Y[:, entering] .> TOL)
    if isempty(pos_idx)
      error("Ilimitado")
    end
    exiting = pos_idx[argmin(t.x_B[pos_idx] ./ t.Y[pos_idx, entering])]
    return entering, exiting
  end

  function pivoting!(t::SimplexTableau, entering::Int, exiting::Int)
    m, n = size(t.Y)
    coef = t.Y[exiting, entering]
    t.Y[exiting, :] /= coef
    t.x_B[exiting] /= coef

    for i in 1:m
      if i != exiting
        c2 = t.Y[i, entering]
        t.Y[i, :] -= c2 * t.Y[exiting, :]
        t.x_B[i] -= c2 * t.x_B[exiting]
      end
    end

    c2 = t.z_c[entering]
    t.z_c .-= c2 * t.Y[exiting, :]'
    t.obj  -= c2 * t.x_B[exiting]

    t.b_idx[exiting] = entering
  end

  function rodar_simplex!(t::SimplexTableau; titulo::String="")
    print_tableau(t, titulo)
    while !is_optimal(t)
      entering, exiting = pivot_point(t)
      println("Pivoteando: entra x_$entering, sai x_$(t.b_idx[exiting])")
      pivoting!(t, entering, exiting)
      print_tableau(t)
    end
  end

  # Recebe (c, A, b, sinais) e devolve a forma padrão estendida com folgas/artificiais.
  # sinais é um vetor de Symbol em {:le, :ge, :eq}.
  function padronizar(c, A, b, sinais)
    m, n = size(A)
    A = Array{Float64}(copy(A))
    b = Array{Float64}(copy(b))
    c = Array{Float64}(copy(c))
    sinais = copy(sinais)

    # Garante b >= 0 multiplicando linhas por -1 quando necessário (e invertendo o sinal).
    for i in 1:m
      if b[i] < 0
        A[i, :] *= -1
        b[i]    *= -1
        sinais[i] = sinais[i] === :le ? :ge :
                    sinais[i] === :ge ? :le : :eq
      end
    end

    # <=  : folga +1                                (básica inicial)
    # >=  : folga -1  + artificial +1               (artificial é a básica inicial)
    # =   :              artificial +1              (artificial é a básica inicial)
    n_folga = count(s -> s === :le || s === :ge, sinais)
    n_art   = count(s -> s === :ge || s === :eq, sinais)
    total = n + n_folga + n_art

    A_ext = zeros(m, total)
    A_ext[:, 1:n] = A

    basis    = zeros(Int, m)
    art_idx  = Int[]
    col_f    = n
    col_a    = n + n_folga

    for i in 1:m
      if sinais[i] === :le
        col_f += 1
        A_ext[i, col_f] = 1.0
        basis[i] = col_f
      elseif sinais[i] === :ge
        col_f += 1
        A_ext[i, col_f] = -1.0
        col_a += 1
        A_ext[i, col_a] = 1.0
        basis[i] = col_a
        push!(art_idx, col_a)
      else  # :eq
        col_a += 1
        A_ext[i, col_a] = 1.0
        basis[i] = col_a
        push!(art_idx, col_a)
      end
    end

    c_ext = zeros(total)
    c_ext[1:n] = c

    return c_ext, A_ext, b, basis, art_idx
  end

  function simplex_duas_fases(c, A, b, sinais)
    n_orig = length(c)
    c_ext, A_ext, b_p, basis, art_idx = padronizar(c, A, b, sinais)
    m, total = size(A_ext)

    println(repeat("=", 64))
    println(" FASE 1 — minimizar a soma das variáveis artificiais")
    println(repeat("=", 64))

    c_p1 = zeros(total)
    c_p1[art_idx] .= 1.0    # min sum w_i

    t = build_tableau(c_p1, A_ext, b_p, basis)
    rodar_simplex!(t; titulo="Tableau inicial (Fase 1)")

    if t.obj > 1e-6
      error("Problema infactível. Mínimo da soma das artificiais = $(t.obj) > 0.")
    end

    # Tenta tirar artificiais que ficaram básicas em valor 0 (degenerescência),
    # pivoteando em alguma coluna não-artificial e não-básica com Y[i,j] != 0.
    for i in 1:m
      if t.b_idx[i] in art_idx
        nb        = setdiff(1:total, t.b_idx)
        nb_no_art = setdiff(nb, art_idx)
        for j in nb_no_art
          if abs(t.Y[i, j]) > TOL
            println("Degenerescência: trocando artificial x_$(t.b_idx[i]) por x_$j na base")
            pivoting!(t, j, i)
            break
          end
        end
      end
    end

    # Linhas em que ainda restou artificial básica são redundantes — removemos.
    rows_drop = [i for i in 1:m if t.b_idx[i] in art_idx]
    rows_keep = setdiff(1:m, rows_drop)
    if !isempty(rows_drop)
      println("Restrições redundantes detectadas e removidas: linhas $rows_drop")
    end

    # Remove colunas artificiais antes de entrar na Fase 2.
    non_art = setdiff(1:total, art_idx)
    col_map = Dict(old => new for (new, old) in enumerate(non_art))
    A_f2     = A_ext[rows_keep, non_art]
    b_f2     = b_p[rows_keep]
    c_f2     = c_ext[non_art]
    basis_f2 = [col_map[t.b_idx[i]] for i in rows_keep]

    println()
    println(repeat("=", 64))
    println(" FASE 2 — otimizar o objetivo original")
    println(repeat("=", 64))

    t2 = build_tableau(c_f2, A_f2, b_f2, basis_f2)
    rodar_simplex!(t2; titulo="Tableau inicial (Fase 2)")

    opt_x = zeros(n_orig)
    for i in 1:length(t2.b_idx)
      if t2.b_idx[i] <= n_orig
        opt_x[t2.b_idx[i]] = t2.x_B[i]
      end
    end
    return opt_x, t2.obj
  end

end
