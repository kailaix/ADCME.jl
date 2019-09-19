# Additional Tools

There are many handy tools implemented in `ADCME` for analysis, benchmarking, input/output, etc. 

```@setup extra
using ADCME
using PyPlot
using LinearAlgebra
sess = Session()
```

## Benchmarking

The functions [`tic`](@ref) and [`toc`](@ref) can be used for recording the runtime between two operations. `tic` starts a timer for performance measurement while `toc` marks the termination of the measurement. Both functions are bound with one operations. For example, we can benchmark the runtime for `svd`

```@example extra
A = constant(rand(10,20))
A = tic(A)
r = svd(A)
B = r.U*diagm(r.S)*r.Vt 
B, t = toc(B)
run(sess, B)
run(sess, t)
```

```@docs
tic
toc
```