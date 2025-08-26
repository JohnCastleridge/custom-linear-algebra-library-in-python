from ..exceptions import (
    IndexOutOfBoundsError,
    InvalidDataError,
    InvalidShapeError,
)

class DunderMixin:
    # === Initialization ===
    def __init__(self, data: list[list[any]]):
        # Validate input matrix structure
        if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
            raise InvalidDataError(obj=data, expected_type='list[list]', operation='Matrix.__init__', reason='Data must be a non‐empty list of lists')
        if any(len(row) != len(data[0]) for row in data):
            raise InvalidShapeError(obj=data, expected_shape=(len(data), len(data[0])), operation='Matrix.__init__', reason='All rows must have the same number of columns')

        self.rows = len(data)
        self.cols = len(data[0])
        self.shape = (self.rows, self.cols)
        self.data = [row[:] for row in data]
    
    # === Indexing & Callable Access ===
    def __getitem__(self, idx: tuple[int]) -> any: # m[i, j]
        if not isinstance(idx, tuple) or not all(isinstance(i, int) for i in idx):
            raise InvalidDataError(idx, 'tuple[int]', operation='Matrix.__getitem__')
        if not len(idx) == 2:
            raise InvalidShapeError(idx, (2,), operation='Matrix.__getitem__')
        i, j = idx
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, i, axis = 'row', operation='Matrix.__getitem__', reason='The first index is out of bounds')
        if (j-1) not in range(self.cols):
            raise IndexOutOfBoundsError(self, j, axis = 'col', operation='Matrix.__getitem__', reason='The second index is out of bounds')
        
        # convert from input 1-based to internal 0-based
        return self.data[i-1][j-1]


    def __setitem__(self, idx: tuple[int], entry: any) -> None: # m[i, j] = entry
        if not isinstance(idx, tuple) or not all(isinstance(i, int) for i in idx):
            raise InvalidDataError(idx, 'tuple[int]', operation='Matrix.__setitem__')
        if not len(idx) == 2:
            raise InvalidShapeError(idx, (2,), operation='Matrix.__getitem__')
        i, j = idx
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, i, axis = 'row', operation='Matrix.__setitem__', reason='The first index is out of bounds')
        if (j-1) not in range(self.cols):
            raise IndexOutOfBoundsError(self, j, axis = 'col', operation='Matrix.__setitem__', reason='The second index is out of bounds')
        
        # convert from input 1-based to internal 0-based
        self.data[i-1][j-1] = entry


    def __call__(self, i, j):
        # allow m(i, j) as alternetiv shorthand, but NB! with 0-indexing
        return self[i+1, j+1]


    # === Representation ===
    def __str__(self):
        decimal_places = 2
        column_padding = 2
        
        if self._is_integer_matrix():
            data = [[str(entry) for entry in row] for row in self.data]
        else:
            data = [[f'{entry:.{decimal_places}f}' for entry in row] for row in self.data]
        
        
        num_len = max(len(string) if '-' not in string else len(string)-1 
                 for row in data for string in row)
        
        data = [[' '*(num_len-len(string)+column_padding) + string 
                 for string in row] for row in data]
        
        return '\n'.join(
            ["┌" + " "*(num_len+2)*self.cols + "  " + "┐"] +
            ["|" + "".join(string for string in row) + "  |" for row in data] +
            ["└" + " "*(num_len+2)*self.cols + "  " + "┘"])

    def __repr__(self):
        pass

    # === Dunder Operations ===
    def __add__(self, other): # +
        if isinstance(other, self.__class__):
            return self.matrix_addition(other)
        return self.scalar_addition(other)
        
    def __sub__(self, other): # -
        return self + -other

    def __mul__(self, other): # *
        if isinstance(other, self.__class__):
            return self.matrix_multiplication(other)
        return self.scalar_multiplication(other)

    def __truediv__(self, other): # /
        if isinstance(other, self.__class__):
            return self * other.inv
        return self * (1/other)

    def __pow__(self, other): # **
        return self.matrix_exponentiation(other)

    def __invert__(self): # ~
        return self.inv

    def __pos__(self): # +self
        return self

    def __neg__(self): # -self
        return self * -1


    # === Dunder NoName ===
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        if not isinstance(other, self.__class__):
            return self.__mul__(other)

    def __rpow__(self, other):
        return self.scalar_exponentiation(other)


    # === Comparison Dunder Methods ===
    def __eq__(self, other): # =
        return self.elementwise_equal(other)

    def __ne__(self, other): # !=
        return self.elementwise_not_equal(other)
    
    def __lt__(self, other): # <
        return self.elementwise_less_than(other)
    
    def __gt__(self, other): # >
        return self.elementwise_greater_than(other)
    
    def __le__(self, other): # <=
        return self.elementwise_less_than_or_equal(other)
    
    def __ge__(self, other): # >=
        return self.elementwise_greater_than_or_equal(other)

    
    def __or__(self, other): # |
        return self.elementwise_OR(other)

    def __and__(self, other): # &&
        return self.elementwise_AND(other)
    
    # === NoName ===
    def __contains__(self, item): # item "in" matrix
        return item in [entry for row in self.data for entry in row]
    
    def __iter__(self): # for entry "in" matrix
        return (entry for row in self.data for entry in row)


