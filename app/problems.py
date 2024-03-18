problems = {

    "1. Linear Algebra": {
        "section": True,
        "description": """## Linear Algebra in Machine Learning and Deep Learning

Linear Algebra is the foundational math behind the algorithms and data analysis in Machine Learning (ML) and Deep Learning (DL). This section is designed to make Linear Algebra concepts accessible, organizing them into practical categories:

### Foundational Skills

- **Get Shape of Matrix**: Understand how data is structured, a critical step in data manipulation.
- **Reshape**: Learn to alter data shapes to fit various ML algorithms' requirements.
- **Matrix Multiplication**: Gain proficiency in this fundamental operation essential for algorithm computations.
- **Mean**: Master the calculation of average values in datasets, a basic statistical tool.
- **Var**: Explore how to calculate variance to understand the distribution of data.

### Intermediate Understanding

- **Covariance Matrix**: Step up your skills by analyzing the relationships between datasets, crucial for understanding data dynamics.

### Advanced Techniques

- **QR Decomposition**: Approach this technique for breaking down matrices, vital for solving complex linear algebra problems.
- **Eigenvalues and Eigenvectors**: Unlock advanced ML capabilities by understanding these concepts, allowing for effective data dimensionality reduction and insights into data's intrinsic properties.

Each section is crafted to build on the last, ensuring a comprehensive understanding from basic operations to advanced analytical methods. Through engaging exercises and clear explanations, you'll be equipped to apply Linear Algebra in practical ML and DL scenarios.
""",
        "example": '',
        "learn": '',
        "starter_code": "",
        "solution": """""",
        "test_cases": [],
    },

    "Get Matrix Shape (easy)": {
        "description": "Write a Python function that gets the shape of a matrix",
        "example": """ Example:
        input: a = [[1,2],[2,4],[4,5]]
        output:(3,2)
        reasoning: There are three rows and two columns """,
        "learn": r'''
        ## Matrix Shape Determination

Consider a general matrix \(M\) consisting of rows and columns:

Matrix \(M\):
$$
M = \begin{pmatrix}
m_{11} & m_{12} & \cdots & m_{1n} \\
m_{21} & m_{22} & \cdots & m_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
m_{m1} & m_{m2} & \cdots & m_{mn}
\end{pmatrix}
$$

The shape of matrix \(M\) is defined by the number of rows and columns it contains, represented as a tuple \((m, n)\), where \(m\) is the number of rows, and \(n\) is the number of columns in the matrix.

**Things to note**: Understanding the shape of a matrix is crucial for matrix operations, including addition, multiplication, and transposition, as these operations often require specific dimensional compatibilities. For example, when multiplying two matrices \(A\) and \(B\) where \(A\) is of shape \((m, n)\) and \(B\) is of shape \((p, q)\), the operation is valid only if \(n = p\), resulting in a new matrix of shape \((m, q)\).
        ''',
        "starter_code": "def get_shape(a:list[list[int|float]])-> set:\n    return (n,m)",
        "solution": """def get_shape(a:list[list[int|float]])-> set:
    return (len(a), len(a[0]))""",
        "test_cases": [
            {"test": "get_shape([[1,2,3],[2,4,5],[6,8,9]])", "expected_output": "(3, 3)"},
            {"test": "get_shape([[1,2,3],[2,4,5]])", "expected_output": "(2, 3)"},
            {"test": "get_shape([[1,2],[2,4],[6,8],[12,4]])", "expected_output": "(4, 2)"},
        ],
    },
    
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
        reasoning: 1*2 + 2*3 = 8;
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
