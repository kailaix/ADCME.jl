using ADCME 
using PyPlot 
using PyCall
mpl = pyimport("tikzplotlib")
db = Database("noise.db")
res = execute(db, """
select noise, kappa from tau 
""")|>collect

noise = [x[1] for x in res]
noise = round.(noise, sigdigits = 2)
kappa = [map(z->parse(Float64, z), split(x[2],',')) for x in res]

close("all")
for k = 1:length(noise)
    plot(1:100:40000, kappa[k][1:100:40000], label = "\$\\sigma_n = $(noise[k])\$")
end
hlines(0.06, 0, 40000, linestyle = "--", color = "k")
legend()
xlabel("Iterations")
ylabel("\$\\tau\$")
grid("on")
minorticks_on()
mpl.save("with_noise.tex")
savefig("with_noise.png")
