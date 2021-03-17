# Visualization with Plotly 

Starting from v0.7.2, [`Plotly`](https://github.com/plotly/plotly.py) plotting backend is included in ADCME. Plotly.py is an open source software for visualizing **interactive** plots based on web technologies. ADCME exposes several APIs of Plotly: [`Plotly`](@ref), [`Layout`](@ref), and common graph objects, such as [`Scatter`](@ref). 

In Plotly, each figure consists of `data`---a collection of traces---and `layout`, whcih is constructed using [`Layout`](@ref). The basic workflow for creating plotly plots is 

- Create a figure object. For example:
```julia
fig = Plot()
```
In the case of subplots within a single figure, use
```julia
fig = Plot(3, 1) # 3 x 1 subplots
```

- Create traces using `add_trace` methods of `fig`. For example
```julia
x = LinRange(0, 2π, 100)
y1 = sin.(x)
y2 = sin.(2*x)
y3 = sin.(3*x)
fig.add_trace(Scatter(x=x, y=y1, name = "Line 1"), row = 1, col = 1)
fig.add_trace(Scatter(x=x, y=y2, name = "Line 2"), row = 2, col = 1)
fig.add_trace(Scatter(x=x, y=y3, name = "Line 3"), row = 3, col = 1)
```

- Update layout or properties
```julia
fig.layout.yaxis.update(title = "Y Axis")
fig.layout.update(hovermode="x unified", showlegend = false)
fig.data[1].update(hovertemplate = """<b>%{y:.2f}</b>""")
```

- Visualize or save to files 
```julia
data = fig.to_dict()
fig.write_html("figure.html", full_html = false) # save to HTML 
fig.show()
```

We can create a figure from dictionary as well. 
```julia
fig = Plot(data)
```