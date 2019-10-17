using Revise
using ADCME
using DelimitedFiles
using LinearAlgebra
using PyPlot
using PyCall

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

ngx = 2
ngy = 2
ngxy = ngx * ngy

bcdof = [ 1 ;2 ;241 ;242] 
bcval = zeros(length(bcdof))

function compute_force(fid)
    force = zeros(sdof)
    if fid==1
        P = 1e5 * 1e5 ;       
        rightedge = findall(coordinates[:,1].==a);
        rightdof = 2*rightedge-ones(Int64,length(rightedge));
        force[rightdof] .= -P*b/(eY+1) ;
        leftedge = findall(coordinates[:,1].==0);
        leftdof = 2*leftedge-ones(Int64, length(leftedge)) ;
        force[leftdof] .= P*b/(eY+1) ;
        return force
    elseif fid==2
        P = 1e5 * 1e5 ;       
        rightedge = findall(coordinates[:,1].==1.);
        rightdof = 2*rightedge-ones(Int64,length(rightedge));
        force[rightdof] .= -P*b/(eY+1) ;
        return force
    elseif fid==3
        P = 1e5 * 1e5 ;       
        leftedge = findall(coordinates[:,1].==0);
        leftdof = 2*leftedge-ones(Int64, length(leftedge)) ;
        force[leftdof] .= P*b/(eY+1) ;
        return force
    elseif fid==4
        P = 1e5 * 1e5 ;       
        topedge = findall(coordinates[:,2].==1.);
        topdof = 2*topedge;
        force[topdof] .= P*b/(eY+1) ;
        return force
    end
end

function constitutive_relation(E::Float64, nu::Float64)
    constant(E/(1-nu^2)*[1 nu 0 ; nu 1 0; 0 0 (1-nu)/2])
end

function constitutive_relation(E::PyObject, nu::PyObject)
    E/(1-nu^2)*tensor([1 nu 0 ; nu 1 0; 0 0 (1-nu)/2])
end

# http://web.mit.edu/course/3/3.11/www/modules/const.pdf page 7
function constitutive_relation(E1::Float64, E2::Float64,
                             nu12::Float64, nu21::Float64, G12::Float64)
    J = [1/E1 -nu21/E2 0
        -nu12/E1 1/E2 0
        0 0 1/G12]
    constant(inv(J))
end

function constitutive_relation(D::PyObject)
    M = [
        1.0 1.0 0.0
        1.0 1.0 0.0
        0.0 0.0 1.0
    ]
    (D'*D).*M
end
