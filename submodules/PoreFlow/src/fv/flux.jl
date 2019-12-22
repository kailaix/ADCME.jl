export tpfa
@doc raw"""
    tpfa(g::TriangleGrid)

Returns the coefficient matrix for two-point flux approximation scheme, assuming zero flux on the boundaries.
    
```math
\int_{\sigma} \nabla u(x) \cdot n(x) ds(x)
```

Here `n(x)` is NOT outward normal vectors, but global `face_normals` in [`TriangleGrid`](@ref).
# References

1. [PorePy](https://github.com/pmgbergen/porepy/blob/develop/src/porepy/numerics/fv/tpfa.py#L15)

2. [Discretization of diffusion fluxes](http://www.scholarpedia.org/article/Finite_volume_method)

3. [Discretization Scheme](https://pangea.stanford.edu/ERE/pdf/pereports/PhD/Moog2013.pdf), Equation (3.31)
"""
function tpfa(g::TriangleGrid)
    ii = Int64[]
    jj = Int64[]
    vv = Float64[]
    for i = 1:g.num_faces
        idx = g.cell_faces[:,i]
        if length(idx.nzind)==1
            continue # the flux is zero 
        else
            p, q = idx.nzind
            d1 = g.cell_centers[p,:] - g.face_centers[i,:]
            d2 = g.cell_centers[q,:] - g.face_centers[i,:]
            n = g.face_normals[i,:]
            T1 = abs(d1'*n)/norm(d1)^2
            T2 = abs(d2'*n)/norm(d2)^2
            sgn = dot(d1, n)>0 ? -1. : 1.; # we derive the equation assuming outward normal
            T = T1*T2/(T1+T2) * sgn
            push!(ii, i, i)
            push!(jj, p, q)
            push!(vv, -T, T)
        end
    end
    return sparse(ii, jj, vv, g.num_faces, g.num_cells)
end