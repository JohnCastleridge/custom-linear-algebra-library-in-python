from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError
)

class UnaryMatrixOperationsMixin:
    def submatrix(self, rows: list[int], cols: list[int]):
        return self.__class__([
             [self[r,c]
              for c in cols] 
              for r in rows
        ])

    def minor(self, rows: list[int], cols: list[int]):
        # must be squre after computing the submatrix 
        #if self.rows - len(rows) != self.cols - len(cols):
            
        return self.submatrix(
            [row for row in range(1, self.rows+1) if row not in rows], 
            [col for col in range(1, self.cols+1) if col not in cols]
        ).det

    def first_minor(self, i: int, j: int):
        return self.minor([i], [j])

    def cofactor(self, i: int, j: int):
        return (-1)**(i+j) * self.M(i, j)

    def determinant(self):
        if not self._is_square():
            raise NotSquareError(self, "determinant")
            
        if self.rows == 1:
            return self[1,1]
        # Laplace expansion
        return sum([self[1,j]*self.C(1,j) for j in range(1, self.cols+1)])
    

    # === Matrix Transformations ===
    def transpose(self):
        return self.__class__([
             [self.data[col][row] 
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def cofactor_matrix(self):
        return self.__class__([
            [self.C(i,j) 
             for j in range(1, self.cols+1)]
             for i in range(1, self.rows+1)
        ])

    def adjugate_matrix(self):
        return self.comatrix.T
    
    def inverse_matrix(self):
        return self.adj * (1/self.det)

    def conjugate_transpose(self):
        "z conjugate = |z|^2 / z"
        return self.__class__([
             [abs(self(row,col))**2 / self(row,col)
              for col in range(self.cols)] 
              for row in range(self.rows)
        ]).T

    def trace(self):
        if not self._is_square():
            raise InvalidDimensionsError(self, 
                operation="trace"
            )
        
        return sum(self.data[i][i] for i in range(self.rows))
    
    # === Property Aliases ===
    T = property(transpose)
    det = property(determinant)
    M = first_minor
    C = cofactor
    comatrix = property(cofactor_matrix)
    adj = property(adjugate_matrix)
    inv = property(inverse_matrix)
    hermitian_transpose = conjugate_transpose
    adjoint = property(conjugate_transpose)
    tr = property(trace)
