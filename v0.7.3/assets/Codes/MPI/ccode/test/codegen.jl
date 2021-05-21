using SymPy 

x,y = @vars x y
u = x*(1-x)*y*(1-y)
ux = diff(u, x)
uy = diff(u, y)
κ = 1+x^2+y^2
f = diff(ux*κ, x) + diff(uy*κ, y)

s = replace(replace(sympy.julia_code(f), ".*"=>"*"), ".^"=>"^")
print(s)