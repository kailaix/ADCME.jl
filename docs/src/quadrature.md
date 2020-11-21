# Numerical Integration


This section describes how to do numerical integration in ADCME. In fact, the numerical integration functionality is independent of automatic differentiation. We can use a third-party library, such as [FastGaussQuadrature](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) for extracts the quadrature weights and points. Then the quadrature weights and points can be used to calculate the integral. 

The general rule for numerical integral is 

$$\int_a^b w(x) f(x) dx = \sum_{i=1}^n w_i f(x_i)$$

The method works best when $f(x)$ can be approximated by a polynomial on the interval $(a, b)$. 

## Examples 

Let us consider some examples. 

### Example 1
$$\int_0^1\frac{\sin(a*x)}{\sqrt{1-x^2}}dx$$

Here $a$ is a tensor (e.g., `a = Variable(1.0)`). We can use the Gauss-Chebyshev quadrature rule of the 1st kind. The corresponding weight function is $\frac{1}{\sqrt{1-x^2}}$

```julia
using FastGaussQuadrature, ADCME
x, w = gausschebyshev(100) # 100 is the number of quadrature nodes
a = Variable(1.0)
integral = sum(cos(a * x) * w)
```

We can verify the result with the exact value 
```julia
sess = Session(); init(sess)
@show sum(cos.(x) .* w), run(sess, integral)
```

The output is 
```
(sum(cos.(x) .* w), run(sess, integral)) = (2.4039394306344133, 2.4039394306344137)
```



### Example 2

$$\int_0^\infty (x-1)^β x^\alpha \exp(-x) dx$$

Here we consider the Gauss-Laguerre quadrature rule. The weight function is $w(x) = x^\alpha \exp(-x)$ and $f(x) = (x-1)^2$.

```julia
using FastGaussQuadrature, ADCME
α = 2.0
x, w = gausslaguerre(100, α)
β = Variable(2.0)
integrand = constant(x .- 1)^β
integral = sum(integrand .* w)
```

We can verify the result with the exact value 
```julia
sess = Session(); init(sess)
@show sum((x .- 1).^2 .* w), run(sess, integral)
```

The output is 
```
(sum((x .- 1) .^ 2 .* w), run(sess, integral)) = (14.000000000000039, 14.000000000000037)
```

The integral rule is also differentiable, for example, we can calculate the gradient of the integral with respect to $\beta$

```julia
run(sess, gradients(integral, β))
```

For convenience, here is a list of supported quadrature rules (run `using FastGaussQuadrature` first)

| Interval            | ω(x)                        | Orthogonal polynomials              | Function |
|---------------------|-----------------------------|-------------------------------------|----------|
| $[−1, 1]$           | 1                           | Legendre polynomials                |   `gausslegendre(n)`       |
| $(−1, 1)$           | $(1-x)^\alpha (1+x)^\beta$  | Jacobi polynomials                  |    `gaussjacobi(n, a, b)`      |
| $(−1, 1)$           | $\frac{1}{sqrt{1-x^2}}$     | Chebyshev polynomials (first kind)  |   `gausschebyshev(n, 1)`       |
| $[−1, 1]$           | $\sqrt{1-x^2}$              | Chebyshev polynomials (second kind) | `gausschebyshev(n, 2)`          |
| $[−1, 1]$           | $\sqrt{(1+x)/(1-x)}$              | Chebyshev polynomials (third kind) | `gausschebyshev(n, 3)`          |
| $[−1, 1]$           | $\sqrt{(1-x)/(1+x)}$              | Chebyshev polynomials (fourth kind) | `gausschebyshev(n, 4)`          |
| $[0, \infty)$       | $e^{-x}$                    | Laguerre polynomials                |   `gausslaguerre(n)`       |
| $[0, \infty)$       | $x^\alpha e^{-x}, \alpha>1$ | Generalized Laguerre polynomials    |      `gausslaguerre(n, α)`    |
| $(-\infty, \infty)$ | $e^{-x^2}$                  | Hermite polynomials                 |    `gausshermite(n)`      |

