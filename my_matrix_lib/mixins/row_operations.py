from ..exceptions import (
    InvalidDataError,
    IndexOutOfBoundsError,
)

class MatrixRowOperationsMixin:
    def row_switching(self, i: int, j: int):
        """
        Rᵢ ↔ Rⱼ
        """
        if not isinstance(i, int):
            raise InvalidDataError(i, 'int', operation='row switching', reason='The first input has not type: "int"')
        if not isinstance(j, int):
            raise InvalidDataError(j, 'int', operation='row switching', reason='The second input has not type: "int"')
        if (i-1) not in range(self.rows) or (j-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, (i, j), operation='row switching', reason='At least one of the indices is out of bounds')

        return self.__class__([
            self.data[i-1] if idx == j-1 else 
            self.data[j-1] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self.data)
        ])

    def row_multiplication(self, i: int, k):
        """
        k⋅Rᵢ → Rᵢ
        """
        if not isinstance(i, int):
            raise InvalidDataError(i, 'int', operation='row multiplication', reason='The index input has an invalid type')
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, i, operation='row multiplication')
        if abs(k) < 1e-8:
            raise ValueError('Can not multiply row with 0')

        return self.__class__([
            [k*a for a in self.data[i-1]] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self.data)
        ])

    def row_addition(self, i: int, j: int, k):
        """
        Rᵢ + k⋅Rⱼ → Rᵢ
        """
        if not isinstance(i, int):
            raise InvalidDataError(i, 'int', operation='row addition', reason='The first input has an invalid type')
        if not isinstance(j, int):
            raise InvalidDataError(j, 'int', operation='row addition', reason='The second input has an invalid type')
        if (i-1) not in range(self.rows) or (j-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, (i, j), operation='row addition')
        
        return self.__class__([
            [self.data[i-1][idx]+k*self.data[j-1][idx] for idx in range(self.cols)] 
            if row == self.data[i-1] else row[:] 
            for row in self.data
        ])
    
    def row_division(self, i: int, k):
        """
        Rᵢ / k → Rᵢ
        """
        return self.row_multiplication(i, 1/k)

    def row_subtraction(self, i: int, j: int, k):
        """
        Rᵢ - k⋅Rⱼ → Rᵢ
        """
        return self.row_addition(i, j, -k)






