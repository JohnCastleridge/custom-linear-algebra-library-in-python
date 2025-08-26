from ..exceptions import (
    InvalidDataError,
    IndexOutOfBoundsError,
)

class MatrixFactoryMixin:
    # === NoName ===
    @classmethod
    def identity(cls, n: int):
        """
        Returns the identity matrix with dimension n×n
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='identity', reason='"n" must be an integer greater then 0')

        return cls([
            [1 if i==j else 0 
            for j in range(n)]
            for i in range(n)
        ])
    
    @classmethod
    def zeros(cls, n, m):
        """
        Returns the zero / null matrix with dimension n×m
        """
        # check if n and m are integers
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='zeros', reason='"n" must be an integer greater then 0')
        if not isinstance(m, int):
            raise InvalidDataError(obj=m, expected_type='int', operation='zeros', reason='"m" must be an integer greater then 0')

        return cls([
            [0 
            for j in range(m)]
            for i in range(n)
        ])
    
    @classmethod
    def ones(cls, n, m):
        """
        Returns the matrix of ones with dimension n×m
        """
        # check if n and m are integers
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='ones', reason='"n" must be an integer greater then 0')
        if not isinstance(m, int):
            raise InvalidDataError(obj=m, expected_type='int', operation='ones', reason='"m" must be an integer greater then 0')
        
        return cls([
            [1
            for j in range(m)]
            for i in range(n)
        ])

    @classmethod
    def exchange(cls, n: int):
        """
        Returns the exchange matrix with dimension n×n
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='exchange', reason='"n" must be an integer greater then 0')

        return cls([
            [1 if i+j==n-1 else 0 
            for j in range(n)]
            for i in range(n)
        ])
    
    @classmethod
    def hilbert(cls, n: int):
        """
        Returns the hilbert  matrix with dimension n×n
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='hilbert', reason='"n" must be an integer greater then 0')

        return cls([
            [1/(i+j-1) 
            for j in range(1, n+1)]
            for i in range(1, n+1)
        ])

    # === NoName ===
    @classmethod
    def matrix_unit(cls, i, j, n, m):
        """
        Returns the matrix unit with dimension n×m and aᵢⱼ = 1
        """

        matrix = cls.__class__.O(n, m)
        matrix[i, j] = 1

        return matrix

    @classmethod
    def diagonal(cls, diagonals: list):
        """
        Returns the square diagonal matrix with the input as diagonal 
        """
        # check if diagonals is an list
        if not isinstance(diagonals, list):
            raise InvalidDataError(obj=diagonals, expected_type='list', operation='diagonal', reason='"diagonals" must be an non-empty list')
        
        return cls([
            [diagonals[i] if i==j else 0
            for j in range(len(diagonals))]
            for i in range(len(diagonals))
        ])
    
    @classmethod
    def vandermonde(cls, x: list):
        """
        Returns the vandermonde matrix
        """
        # check if diagonals is an list
        if not isinstance(x, list):
            raise InvalidDataError(obj=x, expected_type='list', operation='vandermonde', reason='"x" must be an non-empty list')
        
        return cls([
            [x[i]**j
            for j in range(len(x))]
            for i in range(len(x))
        ])
 


    # === Aliases ===
    I = identity
    O = zeros
    J = ones
    H = hilbert
    E = matrix_unit
    D = diagonal
    V = vandermonde