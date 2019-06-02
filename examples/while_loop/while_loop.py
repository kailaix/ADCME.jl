from sympy import *
from sympy.printing import julia_code

x1,x2,x3,y1,y2,y3, D = symbols('x1 x2 x3 y1 y2 y3 D')
M = Matrix([[x1, y1, 1],[x2, y2, 1], [x3, y3, 1]])
T = 0.5*det(M)
A = [[None for _ in range(3)] for _ in range(3)]
A[0][0] = y2-y3
A[0][1] = y3-y1
A[0][2] = y1-y2 
A[1][0] = x3-x2
A[1][1] = x1-x3
A[1][2] = x2-x1 
A[2][0] = x2*y3-x3*y2 
A[2][1] = x3*y1-x1*y3 
A[2][2] = x1*y2-x2*y1
for i in range(3):
    for j in range(3):
        A[i][j] = A[i][j]/det(M)
A = Matrix(A)
print(T.subs(x1, 0.0).subs(y1, 0.0).subs(x2, 0.0).subs(y2,1.0).subs(x3,1.0).subs(y3,0.0))
print(A.subs(x1, 0.0).subs(y1, 0.0).subs(x2, 0.0).subs(y2,1.0).subs(x3,1.0).subs(y3,0.0))
print((A*M).subs(x1, 0.0).subs(y1, 0.0).subs(x2, 0.0).subs(y2,1.0).subs(x3,1.0).subs(y3,0.0))

N = [None for i in range(9)]
for i in range(3):
    for j in range(3):
        N[i+3*j] = D*(A[0,i]*A[0,j]+A[1,i]*A[1,j])
N = Matrix(N)
print(N.subs(x1, 0.0).subs(y1, 0.0).subs(x2, 0.0).subs(y2,1.0).subs(x3,1.0).subs(y3,0.0))
# B = [x.subs(x1, 0.0).subs(y1, 0.0).subs(x2, 0.0).subs(y2,1.0).subs(x3,1.0).subs(y3,0.0) for x in A]
# print(B)
s = [None for i in range(9)]
for i in range(9):
    s[i] = julia_code(N[i])
    s[i] = s[i].replace(".*","*").replace(".^","^").replace("./","/")
s = "T*stack(["+",".join(s)+"])"
print(s)

print(julia_code(T).replace(".*","*").replace(".^","^").replace("./","/"))