export LinearElasticity2D, computeLinearSystem, compute_nonlinear_gradients

fem_op_dict = Dict{String, PyObject}()

function shape_functions(ξ::Float64, η::Float64)
    shape = zeros(4)
    dhdr = zeros(4)
    dhds = zeros(4)
    shape[1]=0.25*(1-ξ)*(1-η);
    shape[2]=0.25*(1+ξ)*(1-η);
    shape[3]=0.25*(1+ξ)*(1+η);
    shape[4]=0.25*(1-ξ)*(1+η);
    dhdr[1]=-0.25*(1-η);
    dhdr[2]=0.25*(1-η);
    dhdr[3]=0.25*(1+η);
    dhdr[4]=-0.25*(1+η);
    dhds[1]=-0.25*(1-ξ);
    dhds[2]=-0.25*(1+ξ);
    dhds[3]=0.25*(1+ξ);
    dhds[4]=0.25*(1-ξ);
    return shape, dhdr, dhds
end

function shape_function_derivatives(n::Int64, dhdr::Array{Float64}, dhds::Array{Float64}, invjacob::Array{Float64})
    dhdx = zeros(n)
    dhdy = zeros(n)
    for i = 1:n 
        dhdx[i]=invjacob[1,1]*dhdr[i]+invjacob[1,2]*dhds[i];
        dhdy[i]=invjacob[2,1]*dhdr[i]+invjacob[2,2]*dhds[i];
    end
    return dhdx, dhdy
end

function fem_jacobian(n::Int64,dhdr::Array{Float64},dhds::Array{Float64},xcoord::Array{Float64},ycoord::Array{Float64})
     jacobian=zeros(2,2)
     for i=1:n
        jacobian[1,1]=jacobian[1,1]+dhdr[i]*xcoord[i]
        jacobian[1,2]=jacobian[1,2]+dhdr[i]*ycoord[i]
        jacobian[2,1]=jacobian[2,1]+dhds[i]*xcoord[i]
        jacobian[2,2]=jacobian[2,2]+dhds[i]*ycoord[i]
     end
     return jacobian
end

function gauss_quadrature(ng::Int64)
    @assert ng == 2
    Gausspoint=zeros(ng);
    Gaussweight=zeros(ng);
       
    Gausspoint[1]=-0.577350269189626;
    Gausspoint[2]=-Gausspoint[1];
    Gaussweight[1]=1.0;
    Gaussweight[2]=Gaussweight[1];
    return Gausspoint,Gaussweight
end

"""
Determines the system dofs for a particular element
`index[i]`: the global index corresponds to global dofs
"""
function element_dof(nd::Array{Int64}, n::Int64, ndof::Int64)
    edof = n * ndof
    index = zeros(Int64, n*ndof)
    k = 0 
    for i = 1:n 
        start = (nd[i]-1)*ndof
        for j = 1:ndof
            k = k + 1
            index[k] = start + j
        end
    end
    index
end

function assemble!(stiffness, k, index)
    edof = length(index)
    for i = 1:edof 
        ii = index[i]
        for j = 1:edof 
            jj = index[j]
            stiffness[ii,jj] += k[i,j]
        end
    end
end

function constraints(stiffness::Array{Float64}, force::Array{Float64}, 
                        bcdof::Array{Int64},bcval::Array{Float64})
    n = length(bcdof)
    sdof = size(stiffness,1)
    for i = 1:n 
        c = bcdof[i]
        for j = 1:sdof 
            stiffness[c, j] = 0.0
        end
        stiffness[c, c] = 1.0
        force[c] = bcval[i]
    end
    return stiffness, force
end

function constraints(stiffness::SparseTensor, force::Array{Float64},
    bcdof::Array{Int64},bcval::Array{Float64})
    local dirichlet_bd
    global fem_op_dict
    if haskey(fem_op_dict, "dirichlet_bd")
        dirichlet_bd = fem_op_dict["dirichlet_bd"]
    else
        compile_op("$(@__DIR__)/CustomOps/DirichletBD/build/libDirichletBd", check=true)
        dirichlet_bd = load_op_and_grad("$(@__DIR__)/CustomOps/DirichletBD/build/libDirichletBd", "dirichlet_bd")
        fem_op_dict["dirichlet_bd"] = dirichlet_bd
    end

    ii, jj, vv = find(stiffness)
    uu = dirichlet_bd(ii, jj, constant(bcdof), vv)
    stiffness = SparseTensor(ii, jj, uu, size(stiffness)...)
    n = length(bcdof)
    for i = 1:n 
        c = bcdof[i]
        force[c] = bcval[i]
    end
    return stiffness, force
end

function fekineps(n::Int64, dhdx::Array{Float64},dhdy::Array{Float64})
    kinmtps = zeros(3, 2*n)
    for i = 1:n 
        i1 = (i-1)*2+1
        i2 = i1 + 1
        kinmtps[1,i1]=dhdx[i];
        kinmtps[2,i2]=dhdy[i];
        kinmtps[3,i1]=dhdy[i];
        kinmtps[3,i2]=dhdx[i];
    end
    kinmtps
end


mutable struct LinearElasticity2D
    ndof::Int64
    coordinates::Array{Float64,2}
    nodes::Array{Int64,2}
    ngx::Int64
    ngy::Int64
    nnel::Int64
    nel::Int64
    Gausspoint::Array{Float64}
    Gaussweight::Array{Float64}
    D::Union{PyObject, Array{Float64}}
    sdof::Int64
    edof::Int64
end

function LinearElasticity2D(coordinates::Array{Float64,2}, nodes::Array{Int64,2},
        D::Union{PyObject, Array{Float64, 2}})
    ndof = 2
    nnel = 4
    nel = size(nodes, 1)
    nnode = size(coordinates, 1)
    sdof = nnode*ndof
    edof = nnel*ndof
    ngx = 2
    ngy = 2
    ngxy = ngx * ngy
    Gausspoint,Gaussweight=gauss_quadrature(ngx)
    LinearElasticity2D(ndof, coordinates, nodes, ngx, ngy, nnel, 
            nel, Gausspoint, Gaussweight, D, sdof, edof)
end

mutable struct Nonlinear2D
    ndof::Int64
    coordinates::Array{Float64,2}
    nodes::Array{Int64,2}
    ngx::Int64
    ngy::Int64
    nnel::Int64
    nel::Int64
    Gausspoint::Array{Float64}
    Gaussweight::Array{Float64}
    constitutive_law::Function
    sdof::Int64
    edof::Int64
end

function Nonlinear2D(coordinates::Array{Float64,2}, nodes::Array{Int64,2},
        constitutive_law::Function)
    ndof = 2
    nnel = 4
    nel = size(nodes, 1)
    nnode = size(coordinates, 1)
    sdof = nnode*ndof
    edof = nnel*ndof
    ngx = 2
    ngy = 2
    ngxy = ngx * ngy
    Gausspoint,Gaussweight=gauss_quadrature(ngx)
    Nonlinear2D(ndof, coordinates, nodes, ngx, ngy, nnel, 
            nel, Gausspoint, Gaussweight, constitutive_law, sdof, edof)
end

function computeLinearSystem(fem::LinearElasticity2D, force::Array{Float64},
    bcdof::Array{Int64},bcval::Array{Float64})
    stiffness = assemble(fem)
    A, force = constraints(stiffness, force, bcdof, bcval)
end

function assemble(fem::LinearElasticity2D)
    local assemble_op
    global fem_op_dict
    if haskey(fem_op_dict, "assemble_op")
        assemble_op = fem_op_dict["assemble_op"]
    else
        compile_op("$(@__DIR__)/CustomOps/Assemble/build/libAssembleOp", check=true)
        assemble_op = load_op_and_grad("$(@__DIR__)/CustomOps/Assemble/build/libAssembleOp",
                             "assemble_op"; multiple=true)
        fem_op_dict["assemble_op"] = assemble_op
    end

    ndof, nel, ngx, ngy, nnel, nodes, coordinates, Gausspoint, Gaussweight, D, sdof, edof =
        fem.ndof, fem.nel, fem.ngx, fem.ngy, fem.nnel, fem.nodes, fem.coordinates, fem.Gausspoint, fem.Gaussweight, fem.D, fem.sdof, fem.edof 
    ngxy = ngx * ngy
    D = convert_to_tensor(fem.D)
    
    # precompute necessary data
    Bs = zeros(nel*ngx*ngy, 3, edof)
    Ws = zeros(nel * ngx * ngy)
    indexes = zeros(Int64, nel, edof)
    for iel=1:nel 
        nd=nodes[iel,:]
        xx = coordinates[nd,1]
        yy = coordinates[nd,2]
        k = zeros(edof,edof)
        for intx=1:ngx
            xi = Gausspoint[intx]
            wtx = Gaussweight[intx]
            for inty=1:ngy
                eta = Gausspoint[inty] 
                wty = Gaussweight[inty]
                shape,dhdr,dhds = shape_functions(xi,eta)
                jacobian = fem_jacobian(nnel,dhdr,dhds,xx,yy)
                detjacob=det(jacobian)
                invjacob=inv(jacobian)
                dhdx,dhdy=shape_function_derivatives(nnel,dhdr,dhds,invjacob)
                B=fekineps(nnel,dhdx,dhdy)
                
                Bs[(iel-1)*ngxy+(intx-1)*ngy+inty, :, :] = B 
                Ws[(iel-1)*ngxy+(intx-1)*ngy+inty] = wtx*wty*detjacob      
                    
            end
        end    
        indexes[iel,:] = element_dof(nd,nnel,ndof)  
    end  
    Bs = constant(Bs)
    Ws = constant(Ws)
    indexes = constant(indexes)

    function condition(i, ta_k)
        i <= nel+1
    end
    function body(i, ta_k)
        k = constant(zeros(edof, edof))
        for intx=1:ngx
            for inty=1:ngy
                p = (i-2)*ngxy+(intx-1)*ngy+inty
                B = tf.gather(Bs, p-1)
                w = Ws[p]
                k = k+B'*D*B*w;
            end
        end 
        ta_k = write(ta_k, i, k)
        i+1, ta_k
    end
    ta_k = TensorArray(nel+1); ta_k = write(ta_k, 1, constant(zeros(edof, edof)))
    i = constant(2, dtype=Int32)
    i, ta_k = while_loop(condition, body, [i, ta_k])
    out = stack(ta_k)
    ii, jj, vv = assemble_op(indexes, out[2:nel+1], constant(sdof))
    SparseTensor(ii,jj,vv,sdof,sdof)
end

function assemble(fem::Nonlinear2D)
    local assemble_op
    global fem_op_dict
    if haskey(fem_op_dict, "assemble_op")
        assemble_op = fem_op_dict["assemble_op"]
    else
        compile_op("$(@__DIR__)/CustomOps/Assemble/build/libAssembleOp", check=true)
        assemble_op = load_op_and_grad("$(@__DIR__)/CustomOps/Assemble/build/libAssembleOp",
                             "assemble_op"; multiple=true)
        fem_op_dict["assemble_op"] = assemble_op
    end

    ndof, nel, ngx, ngy, nnel, nodes, coordinates, Gausspoint, Gaussweight, constitutive_law, sdof, edof =
        fem.ndof, fem.nel, fem.ngx, fem.ngy, fem.nnel, fem.nodes, fem.coordinates, fem.Gausspoint, fem.Gaussweight, fem.constitutive_law, fem.sdof, fem.edof 
    ngxy = ngx * ngy
    
    # precompute necessary data
    Bs = zeros(nel*ngx*ngy, 3, edof)
    Ws = zeros(nel * ngx * ngy)
    indexes = zeros(Int64, nel, edof)
    for iel=1:nel 
        nd=nodes[iel,:]
        xx = coordinates[nd,1]
        yy = coordinates[nd,2]
        k = zeros(edof,edof)
        for intx=1:ngx
            xi = Gausspoint[intx]
            wtx = Gaussweight[intx]
            for inty=1:ngy
                eta = Gausspoint[inty] 
                wty = Gaussweight[inty]
                shape,dhdr,dhds = shape_functions(xi,eta)
                jacobian = fem_jacobian(nnel,dhdr,dhds,xx,yy)
                detjacob=det(jacobian)
                invjacob=inv(jacobian)
                dhdx,dhdy=shape_function_derivatives(nnel,dhdr,dhds,invjacob)
                B=fekineps(nnel,dhdx,dhdy)
                
                Bs[(iel-1)*ngxy+(intx-1)*ngy+inty, :, :] = B 
                Ws[(iel-1)*ngxy+(intx-1)*ngy+inty] = wtx*wty*detjacob      
                    
            end
        end    
        indexes[iel,:] = element_dof(nd,nnel,ndof)  
    end  
    Bs = constant(Bs)
    Ws = constant(Ws)
    indexes = constant(indexes)

    function fun(u)
        strain = compute_strain(Bs, u)
        stress = constitutive_law(strain)
        f_int = compute_internal_force(strain, stress)
        dstress_dstrain = [tf.gradients(stress[:,1], strain)[1] tf.gradients(stress[:,1], strain)[2] tf.gradients(stress[:,1], strain)[3]]
        ii, jj, vv = compute_jacobian(strain, stress, dstress_dstrain)
        J = SparseTensor(ii, jj, vv)
        return f_int, J 
    end
    
    return fun
end


function compute_nonlinear_gradients(loss::PyObject, u::PyObject, 
    θ::PyObject, fun::Function)
    f_int, J = fun(u)
    x = gradients(loss, u)
    δ = tf.stop_gradients(J'\x)
    return -gradients(sum(δ*f_int), θ)
end
