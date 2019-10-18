using Revise
using ADCME
using DelimitedFiles
using LinearAlgebra
using PyPlot

coordinates = readdlm("coordinates.dat")
nodes = readdlm("nodes.dat", Int64)
ndof = 2
nnel = 4
nel = size(nodes, 1)
nnode = size(coordinates, 1)
sdof = nnode*ndof
edof = nnel*ndof
a = 1
b = 1
eX = 10
eY = 10

E = 2.1*10^11;            
nu = 0.3;                       

ngx = 2
ngy = 2
ngxy = ngx * ngy

bcdof = [ 1 2 241 242] 
bcval = zeros(length(bcdof))

force = zeros(sdof)

P = 1e5 * 1e6 ;       
rightedge = findall(coordinates[:,1].==a);
rightdof = 2*rightedge-ones(Int64,length(rightedge));
force[rightdof] .= -P*b/(eY+1) ;
leftedge = findall(coordinates[:,1].==0);
leftdof = 2*leftedge-ones(Int64, length(leftedge)) ;
force[leftdof] .= P*b/(eY+1) ;

Gausspoint,Gaussweight= ADCME.gauss_quadrature(ngx)
D = E/(1-nu^2)*[1 nu 0 ; nu 1 0; 0 0 (1-nu)/2]

stiffness = zeros(sdof, sdof)
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
            shape,dhdr,dhds = ADCME.shape_functions(xi,eta)
            jacobian = ADCME.fem_jacobian(nnel,dhdr,dhds,xx,yy)
            detjacob=det(jacobian)
            invjacob=inv(jacobian)
            dhdx,dhdy=ADCME.shape_function_derivatives(nnel,dhdr,dhds,invjacob)
            B=ADCME.fekineps(nnel,dhdx,dhdy)
       
            k = k+B'*D*B*wtx*wty*detjacob;
        
        end
    end    
    index = ADCME.element_dof(nd,nnel,ndof)
    ADCME.assemble!(stiffness,k,index)
end  

# error()
stiffness,force = ADCME.constraints(stiffness,force,bcdof,bcval);
displacement = stiffness\force 

writedlm("disp.txt", displacement)
xx = Float64[]
yy = Float64[]
zz = Float64[]
ww = Float64[]
for i = 1:nnode
    push!(xx, coordinates[i,1])
    push!(yy, coordinates[i,2])
    push!(zz, displacement[(i-1)*ndof+1])
    push!(ww, displacement[(i-1)*ndof+2])
end
scatter(xx, yy)
scatter(zz+xx, ww+yy)