from matrixlib import Matrix, det, I
from sympy import Symbol, Eq, solve

r = Symbol('r', integer=True)
λ = Symbol('λ')

B = Matrix([[4,11,14],[8,7,-2]])

A = B*B.T
print(A)
P_A = det(A-λ*I(2))

eq  = Eq(P_A, 0)
print('P_A(λ)=', P_A)
solutions = solve(eq, λ)
 

for solution in solutions:
    print(f'λ = {solution}')



