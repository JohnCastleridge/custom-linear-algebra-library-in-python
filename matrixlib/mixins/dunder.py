from typing import Self, Any

from ..exceptions import (
    IndexOutOfBoundsError,
    InvalidDataError,
    InvalidShapeError,
    MatrixValueError,
)

class DunderMixin:
    # === Indexing & Callable Access ===
    def __getitem__(self, key: tuple[int | slice, int | slice]) -> Any | Self: # matrix[key]

        if not isinstance(key, tuple):
            raise InvalidDataError(key, 'tuple[int | slice, int | slice]', operation='Matrix.__getitem__')
        if len(key) != 2:
            raise InvalidShapeError(key, (2,), operation='Matrix.__getitem__', reason='Was not given two positional arguments')
        if not isinstance(key[0], (int, slice)) or not isinstance(key[1], (int, slice)):
            raise InvalidDataError(key, 'tuple[int | slice, int | slice]', operation='Matrix.__getitem__')
        if isinstance(key[0], slice) and not (
            isinstance(key[0].start, (int, type(None))) and 
            isinstance(key[0].stop , (int, type(None))) and 
            isinstance(key[0].step , (int, type(None)))):
            raise MatrixValueError(value=key[0], operation='Matrix.__getitem__', reason='Invalid row slice: start, stop, and step must each be int or None.')
        if isinstance(key[1], slice) and not (
            isinstance(key[1].start, (int, type(None))) and 
            isinstance(key[1].stop , (int, type(None))) and 
            isinstance(key[1].step , (int, type(None)))):
            raise MatrixValueError(value=key[1], operation='Matrix.__getitem__', reason='Invalid column slice: start, stop, and step must each be int or None.')
        
        if isinstance(key[0], int) and isinstance(key[1], int):
            i, j = key
            if i not in range(1, self.rows+1):
                raise IndexOutOfBoundsError(self, i, axis='row', operation='Matrix.__getitem__', reason='The first index is out of bounds')
            if j not in range(1, self.cols+1):
                raise IndexOutOfBoundsError(self, j, axis='col', operation='Matrix.__getitem__', reason='The second index is out of bounds')
            
            return self._data[i-1][j-1] # convert from input 1-based to internal 0-based

        if isinstance(key[0], slice):
            start = key[0].start
            stop  = key[0].stop
            step  = key[0].step
            # Handle empty 
            start = start if start is not None else 1
            stop  = stop  if stop  is not None else self.cols+1
            step  = step  if step  is not None else 1
            # Handle negetive 
            start = start if start > 0 else self.cols+1 + start
            stop  = stop  if stop  > 0 else self.cols+1 + stop
            # list of included row idecis
            rows  = list(range(start, stop, step))
        else: # if key[0] is int
            rows = [key[0]]

        if isinstance(key[1], slice):
            start = key[1].start
            stop  = key[1].stop
            step  = key[1].step
            # Handle empty 
            start = start if start is not None else 1
            stop  = stop  if stop  is not None else self.cols+1
            step  = step  if step  is not None else 1
            # Handle negetive 
            start = start if start > 0 else self.cols+1 + start
            stop  = stop  if stop  > 0 else self.cols+1 + stop
            # list of included cols idecis
            cols  = list(range(start, stop, step))
        else: # if key[1] is int
            cols = [key[1]]

        return self.submatrix(rows, cols)

    def __setitem__(self, idx: tuple[int], entry: any) -> None: # matrix[i, j] = entry
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
        self._data[i-1][j-1] = entry


    def __call__(self, i, j): # to be removed
        # allow m(i, j) as alternetiv shorthand, but NB! with 0-indexing
        return self[i+1, j+1]


    # === Representation ===
    def __str__(self):
        decimal_places = 2
        column_padding = 2
        
        if self._is_integer_matrix():
            data = [[str(entry) for entry in row] for row in self._data]
        elif self._is_floats_matrix():
            data = [[f'{entry:.{decimal_places}f}' for entry in row] for row in self._data]
        else:
            data = [[f'{entry}' for entry in row] for row in self._data]
        
        
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

    # === NoName ===
    def __iter__(self) -> tuple[Any, ...]: # for entry "in" matrix
        return (entry for row in self._data for entry in row)
    
    def __contains__(self, item: Any) -> bool: # item "in" matrix
        return any(entry==item for entry in self)

    # === arithmetic operators ===
    def __add__(self, other: Self | Any) -> Self: # +
        if isinstance(other, self.__class__):
            return self.matrix_addition(other)
        return self.scalar_addition(other)
        
    def __sub__(self, other: Self | Any) -> Self: # -
        return self + -other

    def __mul__(self, other: Self | Any) -> Self: # *
        if isinstance(other, self.__class__):
            return self.matrix_multiplication(other)
        return self.scalar_multiplication(other)

    def __truediv__(self, other: Self | Any) -> Self: # /
        if isinstance(other, self.__class__):
            return self * other.inv
        return self * (1/other)

    def __pow__(self, other: Any) -> Self: # **
        return self.matrix_exponentiation(other)

    # === NoName ===
    def __abs__(self) -> float:
        pass

    def __invert__(self) -> Self: # ~
        return self.inv

    def __pos__(self) -> Self: # +self
        return self

    def __neg__(self) -> Self: # -self
        return self * -1


    # === NoName ===
    def __radd__(self, other: Self | Any) -> Self: # scaler + Matrix
        return self.__add__(other)

    def __rmul__(self, other: Self | Any) -> Self: # scaler * Matrix
        if not isinstance(other, self.__class__):
            return self.__mul__(other)

    def __rpow__(self, other: Any) -> Self: # scaler ** Matrix
        return self.scalar_exponentiation(other)


    # === Comparison Methods ===
    def __eq__(self, other: Self) -> Self: # =
        return self.elementwise_equal(other)

    def __ne__(self, other: Self) -> Self: # !=
        return self.elementwise_not_equal(other)
    
    def __lt__(self, other: Self) -> Self: # <
        return self.elementwise_less_than(other)
    
    def __gt__(self, other: Self) -> Self: # >
        return self.elementwise_greater_than(other)
    
    def __le__(self, other: Self) -> Self: # <=
        return self.elementwise_less_than_or_equal(other)
    
    def __ge__(self, other: Self) -> Self: # >=
        return self.elementwise_greater_than_or_equal(other)

    
    def __or__(self, other: Self) -> Self: # |
        if self._is_boolean_matrix() and other._is_boolean_matrix():
            return self.elementwise_OR(other)
        return self.augment(other)

    def __and__(self, other: Self) -> Self: # &&
        return self.elementwise_AND(other)
