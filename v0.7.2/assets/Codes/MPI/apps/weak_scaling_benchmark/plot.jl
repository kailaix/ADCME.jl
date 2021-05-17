using PyPlot
using DelimitedFiles

R = readdlm("result.txt")
n = sqrt.([1, 4, 9, 16, 25])

close("all")
plot(n, R[1:5, 3], "or-", label = "1 Core")
plot(n[1:4], R[6:9, 3], "xg-", label = "2 Cores")
plot(n[1:3], R[10:12, 3], "^b-", label = "3 Cores")
plot(n, R[1:5, 5], "or--")
plot(n[1:4], R[6:9, 5], "xg--")
plot(n[1:3], R[10:12, 5], "^b--")
legend()
title("Weak Scaling for Forward Computation")
xlabel("\$N\$")
ylabel("Time (sec)")
savefig("forward_weak.png")


close("all")
plot(n, R[1:5, 4], "or-", label = "1 Core")
plot(n[1:4], R[6:9, 4], "xg-", label = "2 Cores")
plot(n[1:3], R[10:12, 4], "^b-", label = "3 Cores")
plot(n, R[1:5, 6], "or--")
plot(n[1:4], R[6:9, 6], "xg--")
plot(n[1:3], R[10:12, 6], "^b--")
legend()
title("Strong Scaling for Gradient Backpropagation")
xlabel("\$N\$")
ylabel("Time (sec)")
savefig("backward_weak.png")