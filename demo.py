from my_matrix_lib import Matrix, I, det, tr, exp, cos, sin

A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
B = Matrix([[1,2,3],[4,5,6]])

real = I(2)
imag = Matrix([[0,-1],[1,0]])


print(exp(real*imag).adjoint)
print(real*cos(real)+imag*sin(real))

print(real*cos(real)+imag*sin(real) == exp(real*imag))

print(det(A))

print(B * A)
