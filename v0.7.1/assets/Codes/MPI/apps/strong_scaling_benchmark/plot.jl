using PyPlot
using DelimitedFiles

R = readdlm("result.txt")
n = [1, 4, 9, 16, 25]

close("all")
loglog(n[end:-1:1], R[1:5, 3], "or-", label = "1 Core")
loglog(n[4:-1:1], R[6:9, 3], "xg-", label = "2 Cores")
loglog(n[3:-1:1], R[10:12, 3], "^b-", label = "3 Cores")
loglog(n[end:-1:1], R[1:5, 5], "or--")
loglog(n[4:-1:1], R[6:9, 5], "xg--")
loglog(n[3:-1:1], R[10:12, 5], "^b--")
loglog(n[end-2:-1:1], 8 ./n[end-2:-1:1], "k--", label = "\$\\mathcal{O}(1/n_{{proc}})\$")
legend()
title("Strong Scaling for Forward Computation")
xlabel("MPI Processors")
ylabel("Time (sec)")
savefig("forward_strong.png")


close("all")
loglog(n[end:-1:1], R[1:5, 4], "or-", label = "1 Core")
loglog(n[4:-1:1], R[6:9, 4], "xg-", label = "2 Cores")
loglog(n[3:-1:1], R[10:12, 4], "^b-", label = "3 Cores")
loglog(n[end:-1:1], R[1:5, 6], "or--")
loglog(n[4:-1:1], R[6:9, 6], "xg--")
loglog(n[3:-1:1], R[10:12, 6], "^b--")
loglog(n[end-2:-1:1], 12 ./n[end-2:-1:1], "k--", label = "\$\\mathcal{O}(1/n_{{proc}})\$")
legend()
title("Strong Scaling for Gradient Backpropagation")
xlabel("MPI Processors")
ylabel("Time (sec)")
savefig("backward_strong.png")
