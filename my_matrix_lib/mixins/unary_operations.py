from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError,
    IndexOutOfBoundsError,
    SingularMatrixError,
    InvalidDataError,
)

class UnaryMatrixOperationsMixin:
    def determinant(self) -> float:
        """
        Returns the determinant of the matrix.
        
        inputs:
        - None
        returns:
        - the determinant of the matrix

        raises:
        - NotSquareError: if the matrix is not square (determinant is only defined for square matrices)
        """
        if not self._is_square():
            raise NotSquareError(self, operation='determinant')
        
        if self.rows == 1:
            return self[1,1]
        
        # Laplace expansion
        return sum([self[1,j]*self.C(1,j) for j in range(1, self.cols+1)])

    def trace(self) -> float:
        """
        Returns the trace of the matrix, which is the sum of the diagonal elements.
        
        inputs:
        - None
        returns:
        - the trace of the matrix

        raises:
        - NotSquareError: if the matrix is not square (trace is only defined for square matrices)
        """
        if not self._is_square():
            raise NotSquareError(self, operation="trace")
        
        return sum(self.data[i][i] for i in range(self.rows))

    # === Matrix Transformations ===
    def transpose(self):
        """
        Returns the transpose of the matrix. 
        
        inputs:
        - None
        returns:
        - a new matrix that is the transpose of the current matrix
        """
        return self.__class__([
             [self.data[col][row] 
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def conjugate_transpose(self):
        """
        Returns the conjugate transpose (or Hermitian transpose) of the matrix.
        
        inputs:
        - None
        returns:
        - a new matrix that is the transpose of the current matrix with each element conjugated
        """
        # z conjugate = |z|^2 / z
        return self.__class__([
             [z-z if abs(z) < 1e-8 else abs(z)*abs(z) / z
              for z in row] 
              for row in self.data
        ]).T

    def submatrix(self, rows: list[int], cols: list[int]):
        """ 
        Returns a submatrix of the current matrix, by only including entries who have row indices containd in "rows" and column indices "cols".

        inputs:
        - rows: a list of row indices (1-based) to include in the submatrix
        - cols: a list of column indices (1-based) to include in the submatrix
        returns:
        - a new matrix that is a submatrix of the current matrix, containing only the specified

        raises:
        - InvalidDataError: if "rows" or "cols" are not lists of integers
        - IndexOutOfBoundsError: if any index in "rows" or "cols" is out of bounds
        - InvalidDimensionsError: if the resulting submatrix is empty (no rows or no columns)
        """
        # check if rows and cols are lists of integers
        if not isinstance(rows, list) or not all(isinstance(i, int) for i in rows) or not rows:
            raise InvalidDataError(rows, 'list[int]', operation='submatrix', reason='"rows" must be a list of integers')
        if not isinstance(cols, list) or not all(isinstance(j, int) for j in cols) or not cols:
            raise InvalidDataError(cols, 'list[int]', operation='submatrix', reason='"cols" must be a list of integers')

        # check if rows and cols are within bounds
        if any(i-1 not in range(self.rows) for i in rows):
            raise IndexOutOfBoundsError(self, rows, axis='row', operation='submatrix', reason='An index in "rows" is out of bounds')
        if any(j-1 not in range(self.cols) for j in cols):
            raise IndexOutOfBoundsError(self, cols, axis='col', operation='submatrix', reason='An index in "cols" is out of bounds')

        # remove duplicates and sort
        rows = sorted(list(set(rows)))
        cols = sorted(list(set(cols)))

        return self.__class__([
             [self[r,c]
              for c in cols] 
              for r in rows
        ])

    def minor(self, rows: list[int], cols: list[int]):
        """
        Returns the determinant of the submatrix defined by excluding the rows and columns specified in "rows" and "cols".

        inputs:
        - rows: a list of row indices (1-based) to exclude from the submatrix
        - cols: a list of column indices (1-based) to exclude from the submatrix
        returns:
        - the determinant of the submatrix formed by excluding the specified rows and columns

        raises:
        - InvalidDataError: if "rows" or "cols" are not lists of integers
        - IndexOutOfBoundsError: if any index in "rows" or "cols" is out of bounds
        - NotSquareError: if the resulting submatrix is not square (determinant is only defined for square matrices)
        - InvalidDimensionsError: if the resulting submatrix is empty (no rows or no columns)
        """
        # check if rows and cols are lists of integers
        if not isinstance(rows, list) or not all(isinstance(i, int) for i in rows) or not rows:
            raise InvalidDataError(rows, 'list[int]', operation='minor', reason='"rows" must be a list of integers')
        if not isinstance(cols, list) or not all(isinstance(j, int) for j in cols) or not cols:
            raise InvalidDataError(rows, 'list[int]', operation='minor', reason='"cols" must be a list of integers')

        # check if rows and cols are within bounds
        if any(i-1 not in range(self.rows) for i in rows):
            raise IndexOutOfBoundsError(self, rows, axis='row', operation='minor', reason='An index in "rows" is out of bounds')
        if any(j-1 not in range(self.cols) for j in cols):
            raise IndexOutOfBoundsError(self, cols, axis='col', operation='minor', reason='An index in "cols" is out of bounds')

        return self.submatrix(
            [row for row in range(1, self.rows+1) if row not in rows], 
            [col for col in range(1, self.cols+1) if col not in cols]
        ).det

    def first_minor(self, i: int, j: int):
        """
        Returns the first minor of the element at position (i, j) in the matrix.

        inputs:
        - i: the row index (1-based) of the element
        - j: the column index (1-based) of the element
        returns:
        - the determinant of the submatrix formed by excluding the i-th row and j-th column

        raises:
        - InvalidDataError: if i or j are not integers
        - IndexOutOfBoundsError: if i or j is out of bounds
        - NotSquareError: if the resulting submatrix is not square (determinant is only defined for square matrices)
        - InvalidDimensionsError: if the resulting submatrix is empty (no rows or no columns
        """
        return self.minor([i], [j])

    def cofactor(self, i: int, j: int):
        """
        Returns the cofactor of the element at position (i, j) in the matrix.

        inputs:
        - i: the row index (1-based) of the element
        - j: the column index (1-based) of the element
        returns:
        - the cofactor of the element at (i, j)

        raises:
        - InvalidDataError: if i or j are not integers
        - IndexOutOfBoundsError: if i or j is out of bounds
        - NotSquareError: if the matrix is not square (cofactor is only defined for square matrices)
        - InvalidDimensionsError: if the matrix is empty (no rows or no columns)
        """
        # check if i and j are integers
        if not isinstance(i, int):
            raise InvalidDataError(i, 'int', operation='cofactor', reason='"i" must be an integer')
        if not isinstance(j, int):
            raise InvalidDataError(j, 'int', operation='cofactor', reason='"j" must be an integer')

        return (-1)**(i+j) * self.M(i, j)

    def cofactor_matrix(self):
        """
        Returns the cofactor matrix of the current matrix.
        
        inputs:
        - None
        returns:
        - a new matrix where each element is the cofactor of the corresponding element in the current matrix

        raises:
        - NotSquareError: if the matrix is not square (cofactor matrix is only defined for square matrices)
        """
        return self.__class__([
            [self.C(i,j) 
             for j in range(1, self.cols+1)]
             for i in range(1, self.rows+1)
        ])

    def adjugate_matrix(self):
        """
        Returns the adjugate (or adjoint) matrix of the current matrix. 
        
        inputs:
        - None
        returns:
        - a new matrix that is the transpose of the cofactor matrix

        raises:
        - NotSquareError: if the matrix is not square (adjugate matrix is only defined for square matrices)
        """
        return self.cofactor_matrix().T
    
    def inverse_matrix(self):
        """
        Returns the inverse of the matrix.
        
        inputs:
        - None
        returns:
        - a new matrix that is the inverse of the current matrix

        raises:
        - singular matrix error: if the matrix is singular (determinant is zero, hence non-invertible)
        - NotSquareError: if the matrix is not square (inverse is only defined for square matrices)
        """
        # check if the inverse exists 
        determinant = self.det
        if abs(determinant) < 1e-8:
            raise SingularMatrixError(self, operation='inverse')
        
        return self.adj * (1/determinant)

    
    # === Property Aliases ===
    T = property(transpose)
    det = property(determinant)
    M = first_minor
    C = cofactor
    comatrix = property(cofactor_matrix)
    adj = property(adjugate_matrix)
    inv = property(inverse_matrix)
    hermitian_transpose = conjugate_transpose
    H = property(conjugate_transpose)
    adjoint = property(conjugate_transpose)
    tr = property(trace)
