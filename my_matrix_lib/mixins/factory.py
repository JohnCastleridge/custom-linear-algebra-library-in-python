from ..exceptions import (
    InvalidDataError,
    MatrixValueError,
)

class MatrixFactoryMixin:
    # === NoName ===
    @classmethod
    def identity(cls, n: int):
        """
        Returns the identity matrix with dimension n×n

        parameters
        ----------
        n : int
            The dimension of the identity matrix to be created.

        Returns
        -------
        Matrix
            The identity matrix with dimension n×n

        Raises
        ------
        InvalidDataError
            If n is not an integer
        MatrixValueError
            If n is less than or equal to 0

        See Also
        --------
        Matrix.I : Alias of this method.
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='identity', reason='"n" must be an integer')
        
        # check if n are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='identity', reason='"n" must be greater than 0')

        return cls([
            [1 if i==j else 0 
            for j in range(n)]
            for i in range(n)
        ])
    
    @classmethod
    def zeros(cls, n, m=None):
        """
        Returns the zero / null matrix with dimension n×m

        parameters
        ----------
        n : int
            The number of rows of the zero matrix to be created.
        m : int, optional
            The number of columns of the zero matrix to be created. If not provided, a square matrix n×n is created.

        Returns
        -------
        Matrix
            The zero matrix with dimension n×m

        Raises
        ------
        InvalidDataError
            If n or m is not an integer
        MatrixValueError
            If n or m is less than or equal to 0

        See Also
        --------
        Matrix.O : Alias of this method.
        """
        # check if n and m are integers
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='zeros', reason='"n" must be an integer')
        if not isinstance(m, int) and m is not None:
            raise InvalidDataError(obj=m, expected_type='int', operation='zeros', reason='"m" must be an integer')

        # check if n and m are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='zeros', reason='"n" must be greater than 0')
        if m is not None or m <= 0:
            raise MatrixValueError(value=m, operation='zeros', reason='"m" must be greater than 0')
        
        # shorthand for square zero matrix
        if m is None:
            m = n

        return cls([
            [0 
            for j in range(m)]
            for i in range(n)
        ])
    
    @classmethod
    def ones(cls, n, m=None):
        """
        Returns the matrix of ones with dimension n×m

        parameters
        ----------
        n : int
            The number of rows of the ones matrix to be created.
        m : int, optional
            The number of columns of the ones matrix to be created. If not provided, a square matrix n×n is created.

        Returns
        -------
        Matrix
            The ones matrix with dimension n×m

        Raises
        ------
        InvalidDataError
            If n or m is not an integer
        MatrixValueError
            If n or m is less than or equal to 0

        See Also
        --------
        Matrix.J : Alias of this method.
        """
        # check if n and m are integers
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='ones', reason='"n" must be an integer')
        if not isinstance(m, int) and m is not None:
            raise InvalidDataError(obj=m, expected_type='int', operation='ones', reason='"m" must be an integer')

        # check if n and m are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='ones', reason='"n" must be greater than 0')
        if m is not None or m <= 0:
            raise MatrixValueError(value=m, operation='ones', reason='"m" must be greater than 0')
        
        # shorthand for square zero matrix
        if m is None:
            m = n
        
        return cls([
            [1
            for j in range(m)]
            for i in range(n)
        ])

    @classmethod
    def exchange(cls, n: int):
        """
        Returns the exchange matrix with dimension n×n

        parameters
        ----------
        n : int
            The dimension of the exchange matrix to be created.

        Returns
        -------
        Matrix
            The exchange matrix with dimension n×n

        Raises
        ------
        InvalidDataError
            If n is not an integer
        MatrixValueError
            If n is less than or equal to 0
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='exchange', reason='"n" must be an integer')
        
        # check if n are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='exchange', reason='"n" must be greater than 0')

        return cls([
            [1 if i+j==n-1 else 0 
            for j in range(n)]
            for i in range(n)
        ])
    
    @classmethod
    def hilbert(cls, n: int):
        """
        Returns the hilbert matrix with dimension n×n
            A Hilbert matrix is a square matrix with entries being the unit fractions
            H(i,j) = 1/(i+j-1)

        parameters
        ----------
        n : int
            The dimension of the hilbert matrix to be created.  

        Returns
        -------
        Matrix
            The hilbert matrix with dimension n×n

        Raises
        ------
        InvalidDataError
            If n is not an integer
        MatrixValueError
            If n is less than or equal to 0

        See Also
        --------
        Matrix.H : Alias of this method.
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='hilbert', reason='"n" must be an integer greater than 0')

        # check if n are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='hilbert', reason='"n" must be greater than 0')

        return cls([
            [1/(i+j-1) 
            for j in range(1, n+1)]
            for i in range(1, n+1)
        ])

    # === NoName ===
    @classmethod
    def matrix_unit(cls, i, j, n, m=None):
        """
        Returns the matrix unit with dimension n×m and aᵢⱼ = 1
        """
        if m is None:
            m = n
        
        matrix = cls.zeros(n, m)
        matrix[i, j] = 1

        return matrix

    @classmethod
    def diagonal(cls, diagonals: list):
        """
        Returns the square diagonal matrix with the input as diagonal 
        """
        # check if diagonals is an list
        if not isinstance(diagonals, list) or not diagonals:
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
        if not isinstance(x, list) or not x:
            raise InvalidDataError(obj=x, expected_type='list', operation='vandermonde', reason='"x" must be an non-empty list')
        
        return cls([
            [x[i]**j
            for j in range(len(x))]
            for i in range(len(x))
        ])
 
     
    @classmethod
    def fourier(cls, n: int, *, imag=complex(0,1), scale=True):
        """
        Returns the fourier matrix
        A Fourier matrix is a scalar multiple of the n-by-n Vandermonde matrix for the roots of unity
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='fourier', reason='"n" must be an integer')
        
        # check if n are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='fourier', reason='"n" must be greater than 0')

        # constants
        e  = 2.71828182845
        pi = 3.14159265359

        # primitive n-th root of unity
        ω = e**(-2*pi*imag / n)

        F = cls.vandermonde([ω**i for i in range(n)])
        if scale:
            F = F * 1/(n**0.5)
        return F
        


    # === Aliases ===
    I = identity
    O = zeros
    J = ones
    H = hilbert
    E = matrix_unit
    D = diagonal
    V = vandermonde
    F = fourier