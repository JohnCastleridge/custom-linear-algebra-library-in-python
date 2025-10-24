from matrixlib import Matrix, det, I
from sympy import Symbol, Eq, solve, symbols

r = Symbol('r', integer=True)
λ = Symbol('λ')

A = Matrix([[1, 3],
            [2, 0]])

P_A = det(A-λ*I(2))

eq  = Eq(P_A, 0)
print('P_A(λ)=', P_A)
solutions = solve(eq, λ)
A, v = symbols('A v', commutative=False)    

for solution in solutions:
    print(f'λ = {solution}')

R = Matrix([[3, -1],
            [2, 1]])

D = Matrix([[3**r,       0],
            [0   , (-2)**r]])

print(5*R*D*~R)
