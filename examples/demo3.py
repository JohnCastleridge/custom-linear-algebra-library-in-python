from matrixlib import Matrix, det, I
from sympy import Symbol, Eq, solve, exp, re


t = Symbol('t')

A = Matrix([[1, 1, 0],
            [-1, 1, 0],
            [0, 0, 1]])


R = Matrix([[0, 1j, -1j],
            [0,  1,  1],
            [1,  0,  0]])

D = Matrix.diagonal([exp(t), exp((1-1j)*t), exp((1+1j)*t)])


B=R*D*~R
B = B.map(lambda entry: float(re(entry.evalf(subs={t: 1}))))
print(B._triple_equal((A).math.exp()))
