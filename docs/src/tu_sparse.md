# Sparse Linear Algebra

ADCME augments TensorFlow APIs by adding sparse linear algebra support. In ADCME, sparse matrices are represented by [`SparseTensor`](@ref). This data structure stores `indices`, `rows` and `cols` of the sparse matrices and keep track of relevant information such as whether it is diagonal for performance consideration. The default is row major (due to TensorFlow backend). 

When evaluating `SparseTensor`, the output will be `SparseMatrixCSC`, the native Julia representation of sparse matrices

```julia
A = run(sess, s) # A has type SparseMatrixCSC{Float64,Int64}
```



## Sparse Matrix Construction

* By passing columns (`Int64`), rows (`Int64`) and values (`Float64`) arrays
```julia
ii = [1;2;3;4]
jj = [1;2;3;4]
vv = [1.0;1.0;1.0;1.0]
s = SparseTensor(ii, jj, vv, 4, 4)
```
* By passing a `SparseMatrixCSC`
```julia
using SparseArrays
s = SparseTensor(sprand(10,10,0.3))
```
* By passing a dense array (tensor or numerical array)
```julia
D = Array(sprand(10,10,0.3)) # a dense array
d = constant(D)
s = dense_to_sparse(d)
```

There are also special constructors. 

| Description                       | Code           |
| --------------------------------- | -------------- |
| Diagonal matrix with diagonal `v` | `spdiag(v)`    |
| Empty matrix with size `m`, `n`   | `spzero(m, n)` |
| Identity matrix with size `m`     | `spdiag(m)`    |



## Matrix Traits

1. Size of the matrices

   ```julia
   size(s) # (10,20)
   size(s,1) # 10
   ```

2. Return `row`, `col`, `val` arrays (also known as COO arrays)

   ```julia
   ii,jj,vv = find(s)
   ```



## Arithmetic Operations

1. Add Subtract

   ```julia
   s = s1 + s2
   s = s1 - s2
   
   ```

2. Scalar Product

   ```julia
   s = 2.0 * s1
   s = s1 / 2.0
   ```

3. Sparse Product

   ```julia
   s = s1 * s2
   ```

4. Transposition

   ```julia
   s = s1'
   ```

## Sparse Solvers

1. Solve a linear system (`s` is a square matrix)

   ```julia
   sol = s\rhs
   ```

2. Solve a least square system (`s` is a tall matrix)

   ```julia
   sol = s\rhs
   ```

!!! note
    The least square solvers are implemented using Eigen sparse linear packages, and the gradients are also implemented. Thus, the following codes will work as expected (the gradients functions will correctly compute the gradients):
    ```julia
    ii = [1;2;3;4]
    jj = [1;2;3;4]
    vv = constant([1.0;1.0;1.0;1.0])
    rhs = constant(rand(4))
    s = SparseTensor(ii, jj, vv, 4, 4)
    sol = s\rhs
    run(sess, sol)
    run(sess, gradients(sum(sol), rhs))
    run(sess, gradients(sum(sol), vv))
    ```

## Assembling Sparse Matrix

In many applications, we want to accumulate `row`, `col` and `val` to assemble a sparse matrix in iterations. For this purpose, we provide the `SparseAssembler` utilities. 

```@docs
SparseAssembler
accumulate
assemble
```

