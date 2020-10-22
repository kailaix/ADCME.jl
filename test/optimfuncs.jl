# test functions for developing optimizers
# see https://en.wikipedia.org/wiki/Test_functions_for_optimization


# optimal = (0, 0, 0, ...)
function sphere(x)
    return sum(x^2)
end

# optimal = (1,1,1,...)
function rosenbrock(x)
    x1 = x[1:end-1]
    x2 = x[2:end]
    return sum(100 * (x2-x1^2)^2 + (1-x1)^2)
end

# optimal = (3, 0.5)
function beale(p)
    x, y = p[1], p[2]
    return (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + 
            (2.625 - x + x*y^3)
end

# optimal = (1, 3)
function booth(p)
    x, y = p[1], p[2]
    return (x + 2y - 7)^2 + (2x+y-5)^2
end

#optimal = (-0.2903, ...)
function stybliski_tang(x)
    return sum(x^4 - 16x^2 + 5x)/2
end

# optimal = (1,1)
function levi(p)
    x, y = p[1], p[2]
    sin(3π*x)^2 + (x-1)^2 * (1+sin(3π*y)^2) + (y-1)^2 * (1+sin(2π*y)^2)
end