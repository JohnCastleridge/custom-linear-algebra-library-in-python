from matrixlib import Matrix, det, I

A = Matrix([[2,2,3],
            [4,5,6],
            [7,8,9]])

B = Matrix([[0,0,0],
            [3,3,10],
            [2,3,4]])

C = Matrix([[0,0,0],
            [0,3,10],
            [0,3,4]])


print(A.RREF._triple_equal(I(3)))