# author: Kailai Xu <kailaix@hotmail.edu>
# time: 12/22/2019
function color_map(extr_value, cmap_type="jet")
    cmap = get_cmap(cmap_type)
    scalar_map = matplotlib.cm.ScalarMappable(cmap=cmap)
    scalar_map.set_array(extr_value)
    scalar_map.set_clim(vmin=extr_value[1], vmax=extr_value[2])
    return scalar_map
end

function PyPlot.:plot(g::Grid; 
    cell_value::Union{Missing,Array{Float64}}=missing,
    vector_value::Union{Missing, Array{Float64}}=missing, kwargs...)
    local ctr
    kwargs = Dict{Symbol, Any}(kwargs)
    figsize = get(kwargs, :figsize, missing)
    if ismissing(figsize)
        figure()
    else
        figure(figsize=figsize)
    end
    xlabel("x")
    ylabel("y")

    
    if !ismissing(cell_value)
        kwargs[:cmap] = get(kwargs, :cmap, (minimum(cell_value), maximum(cell_value)))
        c_ = color_map(kwargs[:cmap]).to_rgba
    end
    

    α = get(kwargs, :alpha, 1)
    linewidth = get(kwargs, :linewidth, 1)
    cells = get(kwargs, :cells, ones(Bool, g.num_cells))
    nodes_ = []
    cval_ = []
    for c = 1:g.num_cells
        if !cells[c]
            continue
        end
        nodes = g.nodes[g.tri[c,:],:]
        push!(nodes_, nodes)
        if !ismissing(cell_value)
            push!(cval_,c_(cell_value[c], α))
        end
    end
    poly = matplotlib.collections.PolyCollection(nodes_, linewidth=linewidth)
    poly.set_edgecolor("k")
    poly.set_facecolor(cval_)
    gca().add_collection(poly)

    if !ismissing(vector_value)
        scale = get(kwargs, :vector_scale, 1)
        n = size(vector_value,1)
        if n == g.num_faces
            ctr = g.face_centers
        elseif n == g.num_faces
            ctr = g.cell_centers
        else
            error("Invalid size of vector_value")
        end
        x_ = zeros(n)
        y_ = zeros(n)
        dx_ = zeros(n)
        dy_ = zeros(n)
        for i = 1:n 
            x_[i] = ctr[i, 1]; dx_[i] = scale * vector_value[i,1]
            y_[i] = ctr[i, 2]; dy_[i] = scale * vector_value[i,2]
        end
        quiver(x_, y_, dx_, dy_)
    end

    if haskey(kwargs, :cmap)
        colorbar(color_map(kwargs[:cmap]))
    end

end

