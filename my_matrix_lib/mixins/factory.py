from math import sqrt, exp, pi

from ..exceptions import (
    InvalidDataError,
    MatrixValueError,
)

class MatrixFactoryMixin:
    @classmethod
    def identity(cls, n: int):
        """
        Return the identity matrix of size n×n.

        Args:
            n (int): The dimension of the identity matrix.

        Returns:
            Matrix: The n×n identity matrix.

        Raises:
            InvalidDataError: If ``n`` is not an integer.
            MatrixValueError: If ``n`` is less than or equal to 0.

        See Also:
            Matrix.I: Alias of this method.
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
        Return a zero (null) matrix of size n×m.

        Args:
            n (int): Number of rows.
            m (int, optional): Number of columns. If ``None``, returns a square n×n matrix.

        Returns:
            Matrix: A matrix of zeros with shape n×m.

        Raises:
            InvalidDataError: If ``n`` or ``m`` (when provided) is not an integer.
            MatrixValueError: If ``n`` is ≤ 0 or ``m`` (when provided) is ≤ 0.

        See Also:
            Matrix.O: Alias of this method.
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
        Return a matrix of ones of size n×m.

        Args:
            n (int): Number of rows.
            m (int, optional): Number of columns. If ``None``, returns a square n×n matrix.

        Returns:
            Matrix: A matrix filled with ones with shape n×m.

        Raises:
            InvalidDataError: If ``n`` or ``m`` (when provided) is not an integer.
            MatrixValueError: If ``n`` is ≤ 0 or ``m`` (when provided) is ≤ 0.

        See Also:
            Matrix.J: Alias of this method.
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
        Return the n×n exchange (anti-identity) matrix.

        The exchange matrix has ones on the anti-diagonal and zeros elsewhere.

        Args:
            n (int): The dimension of the exchange matrix.

        Returns:
            Matrix: The n×n exchange matrix.

        Raises:
            InvalidDataError: If ``n`` is not an integer.
            MatrixValueError: If ``n`` is less than or equal to 0.
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
        Return the n×n Hilbert matrix.

        A Hilbert matrix is a square matrix with entries
        ``H[i, j] = 1 / (i + j − 1)`` for ``i, j`` starting at 1.

        Args:
            n (int): The dimension of the Hilbert matrix.

        Returns:
            Matrix: The n×n Hilbert matrix.

        Raises:
            InvalidDataError: If ``n`` is not an integer.
            MatrixValueError: If ``n`` is less than or equal to 0.

        See Also:
            Matrix.H: Alias of this method.
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

    @classmethod
    def matrix_unit(cls, i, j, n, m=None):
        """
        Return the matrix unit (standard basis matrix) ``E_{ij}`` of size n×m.

        The matrix unit has a single 1 at position ``(i, j)`` and zeros elsewhere.

        Args:
            i (int): Row index (0-based or 1-based depending on the class interface).
            j (int): Column index (0-based or 1-based depending on the class interface).
            n (int): Number of rows.
            m (int, optional): Number of columns. If ``None``, a square n×n matrix is created.

        Returns:
            Matrix: The n×m matrix unit with ``a[i, j] = 1``.

        See Also:
            Matrix.E: Alias of this method.
        """
        if m is None:
            m = n
        
        matrix = cls.zeros(n, m)
        matrix[i, j] = 1

        return matrix

    @classmethod
    def diagonal(cls, diagonals: list):
        """
        Return a square diagonal matrix with the given diagonal entries.

        Args:
            diagonals (list): A non-empty list of diagonal entries.

        Returns:
            Matrix: A square matrix ``D`` where ``D[i, i] = diagonals[i]`` and all off-diagonal
            entries are zero.

        Raises:
            InvalidDataError: If ``diagonals`` is not a non-empty list.

        See Also:
            Matrix.D: Alias of this method.
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
        Return the square Vandermonde matrix built from the samples ``x``.

        The returned matrix ``V`` has entries ``V[i, j] = x[i]**j`` with ``i, j = 0, …, n-1``,
        where ``n = len(x)``.

        Args:
            x (list): A non-empty list of scalar samples.

        Returns:
            Matrix: The n×n Vandermonde matrix.

        Raises:
            InvalidDataError: If ``x`` is not a non-empty list.

        See Also:
            Matrix.V: Alias of this method.
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
        Return the n×n discrete Fourier matrix.

        This is a (scaled) Vandermonde matrix built from the primitive n-th root of unity
        ``ω = exp(-2π·imag / n)``. If ``scale`` is ``True``, the matrix is scaled by ``1/√n``
        to be unitary.

        Args:
            n (int): The dimension of the Fourier matrix.
            imag (complex, keyword-only): Imaginary unit to use (default: ``1j``).
            scale (bool, keyword-only): Whether to apply unitary scaling ``1/√n`` (default: ``True``).

        Returns:
            Matrix: The n×n Fourier matrix (unitary if ``scale=True``).

        Raises:
            InvalidDataError: If ``n`` is not an integer.
            MatrixValueError: If ``n`` is less than or equal to 0.

        See Also:
            Matrix.F: Alias of this method.
        """
        # check if n are integer
        if not isinstance(n, int):
            raise InvalidDataError(obj=n, expected_type='int', operation='fourier', reason='"n" must be an integer')
        
        # check if n are greater than 0
        if n <= 0:
            raise MatrixValueError(value=n, operation='fourier', reason='"n" must be greater than 0')

        # primitive n-th root of unity
        ω = exp(-2*pi*imag / n)

        F = cls.vandermonde([ω**i for i in range(n)])
        if scale: # make unitary
            F = F * 1/sqrt(n)
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
