# author: Kailai Xu <kailaix@hotmail.edu>
# time: 12/22/2019
export TriangleGrid
abstract type Grid end

mutable struct TriangleGrid <: Grid
    nodes::Array{Float64,2}
    faces::Array{Int32,2}
    tri::Array{Int32,2}
    face_nodes::SparseMatrixCSC{Int32,Int64}
    cell_nodes::SparseMatrixCSC{Int32,Int64} 
    cell_faces::SparseMatrixCSC{Int32,Int64}
    cell_volumes::Array{Float64}
    face_length::Array{Float64}
    face_normals::Array{Float64,2}
    num_nodes::Int64
    num_faces::Int64
    num_cells::Int64
    face_centers::Array{Float64,2}
    cell_centers::Array{Float64,2}
end

"""
    make_shared(p::TriangleGrid)

Copy the data in `p` to C++ share memory. 
"""
function make_shared(p::TriangleGrid)

end

function TriangleGrid(p::Array{Float64,2},
    tri::Union{Missing, Array{T, 2}}=missing) where T<:Integer
    if ismissing(tri)
        scipy_spatial = pyimport("scipy.spatial")
        tri = scipy_spatial.Delaunay(p)
        tri = tri.simplices .+ 1
    end
    tri = Int32.(tri)
    num_nodes = size(p, 1)
    num_cells = size(tri,1)
    faces = []

    face_to_cells = Dict{Tuple{Int32, Int32}, Array{Int32}}()
    for i = 1:num_cells
        for (p,q) in [(1,2),(2,3),(3,1)]
            x, y = tri[i,p], tri[i,q]
            push!(faces, [min(x,y) max(x,y)])
            if haskey(face_to_cells, (min(x,y), max(x,y)))
                push!(face_to_cells[(min(x,y), max(x,y))], i)
            else
                face_to_cells[(min(x,y), max(x,y))] = [i]
            end
        end
    end
    faces = vcat(faces...)
    faces = unique(faces, dims=1)
    # faces = sort(faces, dims=1)
    ts = []
    for i = 1:size(faces,1)
        push!(ts, (faces[i,1], faces[i,2]))
    end
    ts = sort(ts)
    for i = 1:size(faces,1)
        faces[i,1] = ts[i][1]
        faces[i,2] = ts[i][2]
    end
    num_faces = size(faces, 1)
    face_nodes = zeros(Int32, num_faces, num_nodes)
    for i = 1:num_faces
        face_nodes[i, faces[i,1]] = 1
        face_nodes[i, faces[i,2]] = 1
    end
    face_nodes = sparse(face_nodes)

    face_normals = zeros(num_faces, 2)
    face_length = zeros(num_faces)
    for i = 1:num_faces
        x = p[faces[i,1],:]
        y = p[faces[i,2],:]
        m = y-x
        face_length[i] = norm(m)
        n = [-m[2];m[1]]
        face_normals[i,:] = n 
    end

    cell_centers = zeros(num_cells, 2)
    face_centers = zeros(num_faces, 2)
    for i = 1:num_cells
        cell_centers[i, :] = mean(p[tri[i,:], :], dims=1)
    end
    for i = 1:num_faces
        face_centers[i, :] = mean(p[faces[i,:],:], dims=1)
    end

    cell_faces = zeros(Int32, num_cells, num_faces)
    for i = 1:num_faces
        for j in face_to_cells[(faces[i,1], faces[i,2])]
            cc = cell_centers[j, :]
            fc = face_centers[i, :]
            if dot(fc-cc, face_normals[i,:])>0
                cell_faces[j, i] = 1
            else
                cell_faces[j, i] = -1
            end
        end
    end
    cell_faces = sparse(cell_faces)

    cell_nodes = zeros(Int32, num_cells, num_nodes)
    for i = 1:num_cells
        cell_nodes[i, tri[i,:]] .= 1
    end
    cell_nodes = sparse(cell_nodes)

    cell_volumes = zeros(num_cells)
    for i = 1:num_cells
        cell_volumes[i] = 1/2*abs(det([ones(3,1) p[tri[i,:],:]]))
    end

    
    TriangleGrid(p,faces,tri,face_nodes,cell_nodes,
            cell_faces, cell_volumes,face_length,
            face_normals,num_nodes,num_faces,num_cells,
            face_centers, cell_centers)
end
