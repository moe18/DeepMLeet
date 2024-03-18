problems = {
    
    "Matrix times Vector (easy)": {
        "description": "Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector",
        "example": """ Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10] 
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10 """,
        "learn": r'''
        ## Matrix Times Vector

        Consider a matrix \(A\) and a vector \(v\), where:

        Matrix \(A\):
        $$
        A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
        $$

        Vector \(v\):
        $$
        v = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}
        $$

        The dot product of \(A\) and \(v\) results in a new vector:
        $$
        A \cdot v = \begin{pmatrix} a_{11}v_1 + a_{12}v_2 \\ a_{21}v_1 + a_{22}v_2 \end{pmatrix}
        $$
        things to note: a n x m matrix will need to be multiplied by a vector of size m or else this would not work.
        ''',
        "starter_code": "def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:\n    return c",
        "solution": """def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    vals = []
    for i in a:
        hold = 0
        for j in range(len(i)):
            hold+=(i[j] * b[j])
        vals.append(hold)

    return vals""",
        "test_cases": [
            {"test": "matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9]],[1,2,3])", "expected_output": "[14, 25, 49]"},
            {"test": "matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9],[12,4,0]],[1,2,3])", "expected_output": "[14, 25, 49, 20]"},
            {"test": "matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3])", "expected_output": "-1"},
        ],
    },
    "Matrix times Matrix (medium)": {
        "description": "multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. C = A dot product B",
        "example": """ 
Example:
        input: A = [[1,2],
                    [2,4]], 
               B = [[2,1],
                    [3,4]]
        output:[[ 8,  9],
                [16, 18]]
        reasoning: 1*1 + 2*3 = 8;
                   2*2 + 3*4 = 16;
                   1*1 + 2*4 = 9;
                   2*1 + 4*4 = 18
                    
Example 2:
        input: A = [[1,2],
                    [2,4]], 
               B = [[2,1],
                    [3,4],
                    [4,5]]
        output: -1
        reasoning: the length of the rows of A does not equal
          the column length of B""",
        "learn": r'''
## Matrix Multiplication

Consider two matrices \(A\) and \(B\), to demonstrate their multiplication, defined as follows:

- Matrix \(A\):
$$
A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
$$

- Matrix \(B\):
$$
B = \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}
$$

The multiplication of matrix \(A\) by matrix \(B\) is calculated as:
$$
A \times B = \begin{pmatrix} a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\ a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22} \end{pmatrix}
$$

This operation results in a new matrix where each element is the result of the dot product between the rows of matrix \(A\) and the columns of matrix \(B\).
''',
        "starter_code": """def matrixmul(a:list[list[int|float]],\n              b:list[list[int|float]])-> list[list[int|float]]: \n return c""",
        "solution": """

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    
    vals = []
    for i in range(len(a)):
        hold = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]
                           
            hold.append(val)
        vals.append(hold)

    return vals""",
        "test_cases": [
            {"test": "matrixmul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]])", "expected_output": "[[26, 20, 14], [38, 29, 20], [74, 56, 38]]"},
            {"test": "matrixmul([[0,0],[2,4],[1,2]],[[0,0],[2,4]])", "expected_output": "[[0, 0], [8, 16], [4, 8]]"},
            {"test": "matrixmul([[0,0],[2,4],[1,2]],[[0,0,1],[2,4,1],[1,2,3]])", "expected_output": "-1"},
        ],
    }
}
