# Custom Optimizer

In this article, we describe how to make your custom optimizer

```@docs
CustomOptimizer
```

We will show here a few examples of custom optimizer. These examples can be cast to your specific applications. 

## Ipopt Custom Optimizer

For a concrete example, let us consider using [Ipopt](https://github.com/JuliaOpt/Ipopt.jl) as a constrained optimization optimizer. 

```julia
using Ipopt
using ADCME

IPOPT = CustomOptimizer() do f, df, c, dc, x0, x_L, x_U
    n_variables = length(x0)
    nz = length(dc(x0)) 
  	m = div(nz, n_variables) # Number of constraints
    g_L, g_U = [-Inf;-Inf], [0.0;0.0]
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            rows[1] = 1; cols[1] = 1
            rows[2] = 1; cols[2] = 1
            rows[3] = 2; cols[3] = 1
            rows[4] = 2; cols[4] = 1
        else
            values[:]=dc(x)
        end
    end
  
  	nele_jac = 0 # Number of non-zeros in Jacobian
    prob = Ipopt.createProblem(n_variables, x_L, x_U, m, g_L, g_U, nz, nele_jac,
            f, (x,g)->(g[:]=c(x)), (x,g)->(g[:]=df(x)), eval_jac_g, nothing)
    addOption(prob, "hessian_approximation", "limited-memory")
    addOption(prob, "max_iter", 100)
  	addOption(prob, "print_level", 2) # 0 -- 15, the larger the number, the more detailed the information

    prob.x = x0
    status = Ipopt.solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])
    println(prob.x)
    prob.x
end

reset_default_graph() # be sure to reset graph before any optimization
x = Variable([1.0;1.0])
x1 = x[1]; x2 = x[2]; 
loss = x2
g = x1
h = x1*x1 + x2*x2 - 1
opt = IPOPT(loss, inequalities=[g], equalities=[h], var_to_bounds=Dict(x=>(-1.0,1.0)))
sess = Session(); init(sess)
minimize(opt, sess)
```

Here is a detailed description of the code

* Typically $\nabla c(x)$ is a $m\times n$ sparse matrix, where $m$ is the number of constraints, $n$ is the number of variables. `nz = length(dc(x0))` computes the number of nonzeros in the Jacobian matrix. 

* `g_L`, `g_U` specify the constraint lower and upper bounds: $g_L \leq c(x) \leq g_U$. If $g_L=g_U=0$, the constraint is reduced to equality constraint. Each of the parameters should have the same length as the number of variables, i.e., $n$

* `eval_jac_g` has two modes. In the `Structure` mode, as we mentioned, the constraint $\nabla c(x)$ is a sparse matrix, and therefore we should specify the nonzero pattern of the sparse matrix in `row` and `col`. However, in our application, we usually assume a dense Jacobian matrix, and therefore, we can always use the following code for `Structure`

  ```julia
  k = 1
  for i = 1:div(nz, n_variables)
    for j = 1:n_variables
      rows[k] = i 
      cols[k] = j
      k += 1
    end
  end
  ```

  For the other mode, `eval_jac_g` simply assign values to the array. 

* We can add optimions to the Ipopt optimizer via `addOptions`. See [here](https://coin-or.github.io/Ipopt/OPTIONS.html) for a full list of available options. 

* To add callbacks, you can simply refactor your functions `f`, `df`, `c`, or `dc`. 