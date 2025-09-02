from my_matrix_lib import Matrix, I, det

A = Matrix([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12]
])
#print(A)

#print(Matrix([[2,3,2],[4,1,7]])*Matrix([[-1,0],[2,3],[-2,1]]))

def f(n,m=None):
    m=n
    print(n+m)

f(1)