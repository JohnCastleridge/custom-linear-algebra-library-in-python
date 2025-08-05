from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError
)


#Exchange matrix: https://en.wikipedia.org/wiki/Exchange_matrix
# https://en.wikipedia.org/wiki/List_of_named_matrices

class MatrixFactoryMixin:
    @classmethod
    def identity(cls, n: int):
        """
        Returns the identity matrix with dimension n⁠×n
        """
        return cls([
            [1 if i==j else 0 
            for j in range(n)]
            for i in range(n)
        ])
    @classmethod
    def zeros(cls, n, m):
        """
        Returns the zero / null matrix with dimension n⁠×m
        """
        return cls([
            [0 
            for j in range(m)]
            for i in range(n)
        ])
    @classmethod
    def ones(cls, n, m):
        """
        Returns the matrix of ones with dimension n⁠×m
        """
        return cls([
            [1
            for j in range(m)]
            for i in range(n)
        ])
    @classmethod
    def diagonal(cls, diagonals: list):
        """
        Returns the square diagonal matrix with the input as diagonal 
        """
        return cls([
            [diagonals[i] if i==j else 0
            for j in range(len(diagonals))]
            for i in range(len(diagonals))
        ])
    @classmethod
    def matrix_unit(cls, i, j, n, m):
        """
        Returns the matrix unit with dimension n⁠×m and aᵢⱼ = 1
        """
        #if i > n or j > m:
        matrix = cls.__class__.O(n, m)
        matrix[i, j] = 1
        return matrix

 
    # === Aliases ===
    I = identity
    O = zeros