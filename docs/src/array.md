# Tensor Operations

| Description                       | API                                             |
| --------------------------------- | ----------------------------------------------- |
| Constant creation                 | `constant(rand(10))`                            |
| Variable creation                 | `Variable(rand(10))`                            |
| Get size                          | `size(x)`                                       |
| Get size of dimension             | `size(x,i)`                                     |
| Get length                        | `length(x)`                                     |
| Resize                            | `reshape(x,5,3)`                                |
| Vector indexing                   | `v[1:3]`,`v[[1;3;4]]`,`v[3:end]`,`v[:]`         |
| Matrix indexing                   | `m[3,:]`, `m[:,3]`, `m[1,3]`,`m[[1;2;5],[2;3]]` |
| Index relative to end             | `v[end]`, `m[1,end]`                            |
| Extract row (most efficient)      | `m[2]`, `m[2,:]`                                |
| Extract column                    | `m[:,3]`                                        |
| Convert to dense diagonal matrix  | `diagm(v)`                                      |
| Convert to sparse diagonal matrix | `spdiag(v)`                                     |
| Extract diagonals as vector       | `diag(m)`                                       |
| Elementwise multiplication        | `a.*b`                                          |
| Matrix (vector) multiplication    | `a*b`                                           |
| Matrix transpose                  | `m'`                                            |
| Dot product                       | `sum(a*b)`                                      |
| Solve                             | `A\b`                                           |
| Inversion                         | `inv(m)`                                        |
| Average all elements              | `mean(x)`                                       |
| Average along dimension           | `mean(x, dims=1)`                               |
| Maximum/Minimum of all elements   | `maximum(x)`, `minimum(x)`                      |
| Squeeze all single dimensions     | `squeeze(x)`                                    |
| Squeeze along dimension           | `squeeze(x, dims=1)`, `squeeze(x, dims=[1;2])`  |
| Reduction (along dimension)       | `norm(a)`, `sum(a, dims=1)`                     |
| Elementwise Multiplication        | `a.*b`                                          |
| Elementwise Power                 | `a^2`                                           |
| SVD                               | `svd(a)`                                        |

