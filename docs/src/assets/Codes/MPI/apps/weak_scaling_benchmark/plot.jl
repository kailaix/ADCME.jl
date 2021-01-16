using DelimitedFiles
using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")

R = readdlm("result.txt")

close("all")
plot(R[1:8,1], R[1:8,3], "r-", label = "1 Core")
plot(R[1:8,1], R[1:8,5], "r--", label = "Hypre Linear Solver (1 Core)")
plot(R[9:12,1], R[9:12,3], "g-", label = "4 Cores")
plot(R[9:12, 1], R[9:12,5], "g--", label = "Hypre Linear Solver (4 Cores)")
xlabel("MPI Processors")
ylabel("Time (sec)")
title("Forward Computation")
savefig("forward_weak.png")

close("all")
plot(R[1:8,1], R[1:8,4], "r-", label = "1 Core")
plot(R[1:8,1], R[1:8,6], "r--", label = "Hypre Linear Solver (1 Core)")
plot(R[9:12,1], R[9:12,4], "g-", label = "4 Cores")
plot(R[9:12, 1], R[9:12,6], "g--", label = "Hypre Linear Solver (4 Cores)")
xlabel("MPI Processors")
ylabel("Time (sec)")
title("Gradient Backpropagation")
savefig("backward_weak.png")