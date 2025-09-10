from ..exceptions import ( 
    NotSquareError,
    IndexOutOfBoundsError,
    SingularMatrixError,
    InvalidDataError,
)

class UnaryMatrixOperationsMixin:
    def determinant(self):
        """
        Return the determinant of the matrix.

        Returns:
            float: The determinant of the matrix.

        Raises:
            NotSquareError: If the matrix is not square (determinant is only defined for square matrices).

        See Also:
            Matrix.det: Alias of this method.

        Notes:
            The determinant is calculated using Laplace expansion along the first row.
        """
        if not self._is_square():
            raise NotSquareError(matrix=self, operation='determinant')
        
        if self.rows == 1:
            return self[1,1]
        
        # Laplace expansion
        return sum(self[1,j]*self.C(1,j) for j in range(1, self.cols+1))

    def trace(self):
        """
        Return the trace of the matrix (sum of diagonal elements).
        
        Returns:
            float: The trace of the matrix.

        Raises:
            NotSquareError: If the matrix is not square (trace is only defined for square matrices).
        
        See Also:
            Matrix.tr: Alias of this method.
        """
        if not self._is_square():
            raise NotSquareError(matrix=self, operation="trace")
        
        return sum(self[i,i] for i in range(1, self.rows+1))

    def transpose(self):
        """
        Return the transpose of the matrix. 
        
        Returns:
            Matrix: A new matrix that is the transpose of the current matrix.

        See Also:
            Matrix.T: Alias of this method.
        """
        return self.__class__([
             [self[row,col]
              for row in range(1, self.rows+1)] 
              for col in range(1, self.cols+1)
        ])

    def hermitian_transpose(self):
        """
        Return the Hermitian (conjugate) transpose of the matrix.
        
        Returns:
            Matrix: A new matrix that is the transpose of the current matrix with each element conjugated.

        See Also:
            Matrix.H: Alias of this method.
        """
        # z conjugate = |z|^2 / z
        return self.__class__([
             [self[i,j]-self[i,j] if abs(self[i,j]) < 1e-8 else abs(self[i,j])*abs(self[i,j]) / self[i,j]
              for j in range(1, self.cols+1)] 
              for i in range(1, self.rows+1)
        ]).T

    def submatrix(self, rows: list[int], cols: list[int]):
        """ 
        Return a submatrix by including entries with row indices in ``rows`` and column indices in ``cols``.

        Args:
            rows (list[int]): A list of row indices (1-based) to include in the submatrix.
            cols (list[int]): A list of column indices (1-based) to include in the submatrix.

        Returns:
            Matrix: A new matrix that is a submatrix of the current matrix, containing only the specified

        Raises:
            InvalidDataError: If ``rows`` or ``cols`` are not lists of integers.
            IndexOutOfBoundsError: If any index in ``rows`` or ``cols`` is out of bounds.
            InvalidDimensionsError: If the resulting submatrix is empty (no rows or no columns).
        """
        # check if rows and cols are lists of integers
        if not isinstance(rows, list) or not all(isinstance(i, int) for i in rows) or not rows:
            raise InvalidDataError(obj=rows, expected_type='list[int]', operation='submatrix', reason='"rows" must be a list of integers')
        if not isinstance(cols, list) or not all(isinstance(j, int) for j in cols) or not cols:
            raise InvalidDataError(obj=cols, expected_type='list[int]', operation='submatrix', reason='"cols" must be a list of integers')

        # check if rows and cols are within bounds
        if any(i-1 not in range(self.rows) for i in rows):
            raise IndexOutOfBoundsError(matrix=self, index=rows, axis='row', operation='submatrix', reason='An index in "rows" is out of bounds')
        if any(j-1 not in range(self.cols) for j in cols):
            raise IndexOutOfBoundsError(matrix=self, index=cols, axis='col', operation='submatrix', reason='An index in "cols" is out of bounds')

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
        Return the determinant of the submatrix defined by excluding the rows and columns in ``rows`` and ``cols``.

        Args:
            rows (list[int]): A list of row indices (1-based) to exclude from the submatrix.
            cols (list[int]): A list of column indices (1-based) to exclude from the submatrix.

        Returns:
            float: The determinant of the submatrix formed by excluding the specified rows and columns.

        Raises:
            InvalidDataError: If ``rows`` or ``cols`` are not lists of integers.
            IndexOutOfBoundsError: If any index in ``rows`` or ``cols`` is out of bounds.
            NotSquareError: If the resulting submatrix is not square (determinant is only defined for square matrices).
            InvalidDimensionsError: If the resulting submatrix is empty (no rows or no columns).
        """
        # check if rows and cols are lists of integers
        if not isinstance(rows, list) or not all(isinstance(i, int) for i in rows) or not rows:
            raise InvalidDataError(matrix=rows, expected_type='list[int]', operation='minor', reason='"rows" must be a list of integers')
        if not isinstance(cols, list) or not all(isinstance(j, int) for j in cols) or not cols:
            raise InvalidDataError(matrix=cols, expected_type='list[int]', operation='minor', reason='"cols" must be a list of integers')

        # check if rows and cols are within bounds
        if any(i-1 not in range(self.rows) for i in rows):
            raise IndexOutOfBoundsError(matrix=self, index=rows, axis='row', operation='minor', reason='An index in "rows" is out of bounds')
        if any(j-1 not in range(self.cols) for j in cols):
            raise IndexOutOfBoundsError(matrix=self, index=cols, axis='col', operation='minor', reason='An index in "cols" is out of bounds')

        return self.submatrix(
            [row for row in range(1, self.rows+1) if row not in rows], 
            [col for col in range(1, self.cols+1) if col not in cols]
        ).det

    def first_minor(self, i: int, j: int):
        """
        Return the first minor of the element at position (i, j).

        Args:
            i (int): The row index (1-based) of the element.
            j (int): The column index (1-based) of the element.
        
        Returns:
            float: The determinant of the submatrix formed by excluding the i-th row and j-th column.

        Raises:
            InvalidDataError: If ``i`` or ``j`` are not integers.
            IndexOutOfBoundsError: If ``i`` or ``j`` is out of bounds.
            NotSquareError: If the resulting submatrix is not square (determinant is only defined for square matrices).
            InvalidDimensionsError: If the resulting submatrix is empty (no rows or no columns).
        
        See Also:
            Matrix.M: Alias of this method.
        """
        return self.minor([i], [j])

    def cofactor(self, i: int, j: int):
        """
        Return the cofactor of the element at position (i, j). Defined as
        C(i, j) = (-1)^(i+j) * M(i, j), where M(i, j) is the first minor of the element at (i, j).

        Args:
            i (int): The row index (1-based) of the element.
            j (int): The column index (1-based) of the element.

        Returns:
            float: The cofactor of the element at (i, j).

        Raises:
            InvalidDataError: If ``i`` or ``j`` are not integers.
            IndexOutOfBoundsError: If ``i`` or ``j`` is out of bounds.
            NotSquareError: If the matrix is not square (cofactor is only defined for square matrices).
            InvalidDimensionsError: If the matrix is empty (no rows or no columns).
        
        See Also:
            Matrix.C: Alias of this method.
        """
        # check if i and j are integers
        if not isinstance(i, int):
            raise InvalidDataError(obj=i, expected_type='int', operation='cofactor', reason='"i" must be an integer')
        if not isinstance(j, int):
            raise InvalidDataError(obj=j, expected_type='int', operation='cofactor', reason='"j" must be an integer')

        return (-1)**(i+j) * self.M(i, j)

    def cofactor_matrix(self):
        """
        Return the cofactor matrix of the current matrix. Defined as a matrix where each element
        is the cofactor of the corresponding element in the current matrix.
        
        Returns:
            Matrix: A new matrix where each element is the cofactor of the corresponding element in the current matrix.

        Raises:
            NotSquareError: If the matrix is not square (cofactor matrix is only defined for square matrices).
        
        See Also:
            Matrix.comatrix: Alias of this method.
        """
        return self.__class__([
            [self.C(i,j) 
             for j in range(1, self.cols+1)]
             for i in range(1, self.rows+1)
        ])

    def adjugate_matrix(self):
        """
        Return the adjugate (adjoint) matrix of the current matrix. Defined as the transpose of the cofactor matrix.
        
        Returns:
            Matrix: A new matrix that is the transpose of the cofactor matrix.

        Raises:
            NotSquareError: If the matrix is not square (adjugate matrix is only defined for square matrices).
        
        See Also:
            Matrix.adj: Alias of this method.
        """
        return self.cofactor_matrix().T
    
    def inverse_matrix(self):
        """
        Return the inverse of the matrix.
        
        Returns:
            Matrix: A new matrix that is the inverse of the current matrix.

        Raises:
            SingularMatrixError: If the matrix is singular (determinant is zero, hence non-invertible).
            NotSquareError: If the matrix is not square (inverse is only defined for square matrices).
        
        See Also:
            Matrix.inv: Alias of this method.
        """
        # check if the inverse exists 
        determinant = self.det
        if abs(determinant) < 1e-8:
            raise SingularMatrixError(matrix=self, operation='inverse')
        
        return self.adj * (1/determinant)

    
    # === Aliases ===
    det = property(determinant)
    tr = property(trace)
    T = property(transpose)
    H = property(hermitian_transpose)
    M = first_minor
    C = cofactor
    comatrix = property(cofactor_matrix)
    adj = property(adjugate_matrix)
    inv = property(inverse_matrix)
