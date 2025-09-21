from custom_matrix_lib_py.core import Matrix


def test_matrix_multiplication():
    A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
    B = Matrix([[4,5,1],[6,7,8],[2,3,4]])
    expected = Matrix([[22,28,29],[58,73,68], [94,118,107]]).data
    computed = A.matrix_multiplication(B).data
    success = computed==expected
    assert success, "_matrix_multiplication failed"
