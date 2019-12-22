using Revise
using ADCME
using PyPlot
matplotlib.use("agg")
p = rand(10,2)

g = RectangleMesh(10,10)
plot(g); savefig("frame.png")
plot(g; cell_value=rand(g.num_cells)); savefig("cell.png")
plot(g; vector_value=g.face_normals); savefig("normal.png")