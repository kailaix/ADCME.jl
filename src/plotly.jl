for o in [:Scatter, :Bar, :Scattergl, :Pie, :Heatmap, :Image, :Contour, :Table, :Box, :Violin, 
            :Histogram, :Histogram2dContour, :Ohlc, :Candlestick, :Waterfall, :Funnel, :Funnelarea, 
            :Indicator, :Scatter3d, :Surface, :Mesh3d, :Cone, :Volume, :Isosurface, :Scattergeo, 
            :Choropleth, :Scattermapbox, :Choroplethmapbox, :Densitymapbox,
            :Scatterpolar, :Scatterpolargl, :Barpolar, :Scatterternary, :Sunburst, :Treemap, :Sankey,
            :Splom, :Parcats, :Parcoords, :Carpet, :Scattercarpet, :Contourcarpet, :Layout]
    @eval begin
        export $o
        function $o(args...;kwargs...)
            plotly.graph_objects.$o(args...;kwargs...)
        end
    end
end

export Plot
"""
    Plot(rows::Int64 = 1, cols::Int64 = 1, args...; kwargs...)

Makes a figure consists of `rows × cols` subplots. 

# Example 
```julia 
fig = Plot(3,1)
x = LinRange(0,1,100)
y1 = sin.(x)
y2 = sin.(2*x)
y3 = sin.(3*x)
fig.add_trace(Scatter(x=x, y=y1, name = "Line 1"), row = 1, col = 1)
fig.add_trace(Scatter(x=x, y=y2, name = "Line 2"), row = 2, col = 1)
fig.add_trace(Scatter(x=x, y=y3, name = "Line 3"), row = 3, col = 1)
fig.show()
```
"""
function Plot(rows::Int64 = 1, cols::Int64 = 1, args...; kwargs...)
    spt = pyimport("plotly.subplots")
    spt.make_subplots(rows = rows, cols = cols, args...;kwargs...)
end

export to_html
"""
    to_html(fig::PyObject, filename::String, args...;
    include_plotlyjs = "cnd",
    include_mathjax = "cnd",
    full_html = true,
    kwargs...)

Exports the figure `fig` to an HTML file. 
"""
function to_html(fig::PyObject, filename::String, args...;
        include_plotlyjs = "cnd",
        include_mathjax = "cnd",
        full_html = true,
        kwargs...)
    open(filename, "w") do io 
        cnt = fig.to_html(
            include_plotlyjs = include_plotlyjs,
            include_mathjax = include_mathjax, 
            full_html = full_html
        )
        write(io, cnt)
    end
end