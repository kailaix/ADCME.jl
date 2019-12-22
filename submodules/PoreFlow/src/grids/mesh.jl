export RectangleMesh
function RectangleMesh(nx = 10, ny = 10, x0 = 0.0, y0 = 0.0, x1 = 1.0, y1 = 1.0)
    p = zeros((nx+1)*(ny+1), 2)
    tri = zeros(Int32, nx*ny*2, 3)
    hx = 1/nx; hy = 1/ny;
    k = 1
    for j = 1:ny+1
        for i = 1:nx+1
            p[k, :] = [hx*(i-1) hy*(j-1)]
            k += 1
        end
    end
    k = 1
    for j = 1:ny
    for i = 1:nx 
         
            tri[k, :] = [(i-1)+(j-1)*(nx+1) i+(j-1)*(nx+1) i+j*(nx+1)]
            tri[k+1, :] = [i-1+(j-1)*(nx+1) i-1+j*(nx+1) i+j*(nx+1)]
            k += 2
        end
    end
    tri = tri .+ 1
    p[:,1] = x0 .+ p[:,1]*(x1-x0)
    p[:,2] = y0 .+ p[:,2]*(y1-y0)
    TriangleGrid(p, tri)
end