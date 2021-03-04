using ADCME 
using PyPlot
db = Database("hessian.db")

close("all")
plot(npoints, npoints, "k--")

for seed in [2, 23, 233]
    c = execute(db, """
    SELECT nlambda, npoints from eigvals where seed=$seed order by npoints 
    """)
    vals = collect(c)
    global npoints = [x[2] for x in vals]
    vals = [x[1] for x in vals]
    plot(npoints, vals, ".-", label = "Seed = $seed")
end 
legend()
xlabel("\$||\\mathcal{I}||\$")
ylabel("Positive Eigenvalues")
ylim(0, 15)
grid("on")
minorticks_on()
savefig("hessian_eigenvalue_rank.png")