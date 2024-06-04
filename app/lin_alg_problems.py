problems = {

    "1. Linear Algebra": {
        "section": True,
        "description": """## Linear Algebra in Machine Learning and Deep Learning

Linear Algebra is crucial for understanding and working with data in Machine Learning (ML) and Deep Learning (DL). This section introduces Linear Algebra concepts that are fundamental for ML and DL, providing a structured approach to mastering this essential mathematical foundation:

### Foundational Skills

- **Get Matrix Shape**: Knowing the dimensions of data matrices is essential for preprocessing and model architecture design.
- **Matrix Multiplication**: Crucial for forward and backward passes in neural networks, filters in convolutional networks, and more.
- **Scalar and Element-wise Operations**: Understand how to manipulate matrices and vectors for feature scaling and normalization.
- **Determinants and Inverses**: Learn about matrix properties that are vital for understanding linear transformations and solving systems of linear equations.
- **Eigenvalues and Eigenvectors**: Discover how these concepts are used in dimensionality reduction techniques such as PCA, and in understanding the stability of systems.

### Intermediate Understanding

- **Singular Value Decomposition (SVD)**: A key factorization method for solving linear systems, data compression, and noise reduction.
- **Covariance Matrices**: Essential for understanding the variance within datasets and for feature selection.

### Advanced Techniques

- **Linear Transformations**: Dive into the geometric interpretations of matrix operations and their applications in ML and DL.
- **Orthogonality and Orthonormality**: Concepts critical for understanding the stability of numerical methods and for reducing overfitting through regularization.

Each topic is designed to build upon the previous, ensuring a deep and comprehensive understanding of how Linear Algebra powers machine learning algorithms and data analysis. Practical exercises and examples will demonstrate the application of these mathematical concepts in real-world ML and DL scenarios.""",
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
        "video":"https://youtu.be/eQWOcMiT7ks?si=xu5pD44Tq2wZOxgO",
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
        "video":"https://youtu.be/DNoLs5tTGAw?si=vpkPobZMA8YY10WY",
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
    },"Transpose of a Matrix (easy)": {
    "description": "Write a Python function that computes the transpose of a given matrix.",
    "example": """Example:
        input: a = [[1,2,3],[4,5,6]]
        output: [[1,4],[2,5],[3,6]]
        reasoning: The transpose of a matrix is obtained by flipping rows and columns.""",
    "video": "https://youtu.be/fj0ZJ2gTSmI?si=vG8VSqASjyG0eNLY",
    "learn": r'''
        ## Transpose of a Matrix

        Consider a matrix \(M\) and its transpose \(M^T\), where:

        Original Matrix \(M\):
        $$
        M = \begin{pmatrix} a & b & c \\ d & e & f \end{pmatrix}
        $$

        Transposed Matrix \(M^T\):
        $$
        M^T = \begin{pmatrix} a & d \\ b & e \\ c & f \end{pmatrix}
        $$

        Transposing a matrix involves converting its rows into columns and vice versa. This operation is fundamental in linear algebra for various computations and transformations.
        ''',
    "starter_code": "def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:\n    return b",
    "solution": """def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]""",
    "test_cases": [
        {"test": "transpose_matrix([[1,2],[3,4],[5,6]])", "expected_output": "[[1, 3, 5], [2, 4, 6]]"},
        {"test": "transpose_matrix([[1,2,3],[4,5,6]])", "expected_output": "[[1, 4], [2, 5], [3, 6]]"}
    ]
}, "Calculate Mean by Row or Column (easy)":{
    "description": "Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.",
    "example": """Example:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
        output: [4.0, 5.0, 6.0]
        reasoning: Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].
        
        Example 2:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'
        output: [2.0, 5.0, 8.0]
        reasoning: Calculating the mean of each row results in [(1+2+3)/3, (4+5+6)/3, (7+8+9)/3].""",
        "video":'https://youtu.be/OTm0yijNYKc',
    "learn": r'''
        ## Calculate Mean by Row or Column

        Calculating the mean of a matrix by row or column involves averaging the elements across the specified dimension. This operation provides insights into the distribution of values within the dataset, useful for data normalization and scaling.

        ### Row Mean
        The mean of a row is computed by summing all elements in the row and dividing by the number of elements. For row \(i\), the mean is:
        $$
        \mu_{\text{row } i} = \frac{1}{n} \sum_{j=1}^{n} a_{ij}
        $$
        where \(a_{ij}\) is the matrix element in the \(i^{th}\) row and \(j^{th}\) column, and \(n\) is the total number of columns.

        ### Column Mean
        Similarly, the mean of a column is found by summing all elements in the column and dividing by the number of elements. For column \(j\), the mean is:
        $$
        \mu_{\text{column } j} = \frac{1}{m} \sum_{i=1}^{m} a_{ij}
        $$
        where \(m\) is the total number of rows.

        This mathematical formulation helps in understanding how data is aggregated across different dimensions, a critical step in various data preprocessing techniques.
        ''',
    "starter_code": "def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:\n    return means",
    "solution": """def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")""",
    "test_cases": [
        {
            "test": "calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column')",
            "expected_output": "[4.0, 5.0, 6.0]"
        },
        {
            "test": "calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'row')",
            "expected_output": "[2.0, 5.0, 8.0]"
        }
    ]
}

,
    "Scalar Multiplication of a Matrix (easy)":{
    "description": "Write a Python function that multiplies a matrix by a scalar and returns the result.",
    "example": """Example:
        input: matrix = [[1, 2], [3, 4]], scalar = 2
        output: [[2, 4], [6, 8]]
        reasoning: Each element of the matrix is multiplied by the scalar.""",
    "video": "https://youtu.be/iE2NvpvZRBk",
    "learn": r'''
        ## Scalar Multiplication of a Matrix

        When a matrix \(A\) is multiplied by a scalar \(k\), the operation is defined as multiplying each element of \(A\) by \(k\). 

        Given a matrix \(A\):
        $$
        A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
        $$

        And a scalar \(k\), the result of the scalar multiplication \(kA\) is:
        $$
        kA = \begin{pmatrix} ka_{11} & ka_{12} \\ ka_{21} & ka_{22} \end{pmatrix}
        $$

        This operation scales the matrix by \(k\) without changing its dimension or the relative proportion of its elements.
        ''',
    "starter_code": "def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:\n    return result",
    "solution": """def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]""",
    "test_cases": [
        {"test": "scalar_multiply([[1,2],[3,4]], 2)", "expected_output": "[[2, 4], [6, 8]]"},
        {"test": "scalar_multiply([[0,-1],[1,0]], -1)", "expected_output": "[[0, 1], [-1, 0]]"}
    ]
}
,
"Calculate Eigenvalues of a Matrix (medium)": {
    "description": """Write a Python function that calculates the eigenvalues of a 2x2 matrix.
      The function should return a list containing the eigenvalues, sort values from highest to lowest.""",
    "example": """Example:
        input: matrix = [[2, 1], [1, 2]]
        output: [3.0, 1.0]
        reasoning: The eigenvalues of the matrix are calculated using the characteristic equation of the matrix,
          which for a 2x2 matrix is λ^2 - trace(A)λ + det(A) = 0, where λ are the eigenvalues.""",
    "video": 'https://youtu.be/AMCFzIaHc4Y',
    "learn": r'''## Calculate Eigenvalues

Eigenvalues of a matrix offer significant insight into the matrix's behavior, particularly in the context of linear transformations and systems of linear equations.

### Definition
For a square matrix \(A\), eigenvalues are scalars \(\lambda\) that satisfy the equation for some non-zero vector \(v\) (eigenvector):
$$
Av = \lambda v
$$

### Calculation for a 2x2 Matrix
The eigenvalues of a 2x2 matrix \(A\), given by:
$$
A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
$$
are determined by solving the characteristic equation:
$$
\det(A - \lambda I) = 0
$$
This simplifies to a quadratic equation:
$$
\lambda^2 - \text{tr}(A) \lambda + \det(A) = 0
$$
Here, the trace of A, denoted as tr(A), is a + d, and the determinant of A, denoted as det(A), is ad - bc. Solving this equation yields the eigenvalues, λ.

### Significance
Understanding eigenvalues is essential for analyzing the effects of linear transformations represented by the matrix. They are crucial in various applications, including stability analysis, vibration analysis, and Principal Component Analysis (PCA) in machine learning.
''',
    "starter_code": "def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:\n    return eigenvalues",
    "solution": """def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]""",
    "test_cases": [
        {
            "test": "calculate_eigenvalues([[2, 1], [1, 2]])",
            "expected_output": "[3.0, 1.0]"
        },
        {
            "test": "calculate_eigenvalues([[4, -2], [1, 1]])",
            "expected_output": "[3.0, 2.0]"
        }
    ]
},
    "Calculate 2x2 Matrix Inverse (medium)": {
        "description": "Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.",
        "example": """Example:
            input: matrix = [[4, 7], [2, 6]]
            output: [[0.6, -0.7], [-0.2, 0.4]]
            reasoning: The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.""",
        "video": "",
        "learn": r'''
            ## Calculating the Inverse of a 2x2 Matrix

            The inverse of a matrix A is another matrix, often denoted A^-1, such that:

            $$
            AA^-1 = A^-1A = I
            $$

            where I is the identity matrix. For a 2x2 matrix:
            $$
            A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
            $$

            The inverse is:
            $$
            A^-1 = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
            $$

            provided that the determinant $$\det(A) = ad - bc$$ is non-zero. If $$det(A) = 0 $$, the matrix does not have an inverse.

            This process is critical in many applications including solving systems of linear equations, where the inverse is used to find solutions efficiently.
            ''',
        "starter_code": "def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:\n    return inverse",
        "solution": """def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse""",
        "test_cases": [
            {"test": "inverse_2x2([[4, 7], [2, 6]])", "expected_output": "[[0.6, -0.7], [-0.2, 0.4]]"},
            {"test": "inverse_2x2([[1, 2], [2, 4]])", "expected_output": "None"},
            {"test": "inverse_2x2([[2, 1], [6, 2]])", "expected_output": "[[-1.0, 0.5], [3.0, -1.0]]"}
        ]
    }

,
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
        "video":'https://youtu.be/N2j0fA2E9k4',
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
    },
    "Calculate Covariance Matrix (medium)":{
    "description": "Write a Python function that calculates the covariance matrix from a list of vectors. Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.",
    "example": """Example:
        input: vectors = [[1, 2, 3], [4, 5, 6]]
        output: [[1.0, 1.0], [1.0, 1.0]]
        reasoning: The dataset has two features with three observations each. The covariance between each pair of features (including covariance with itself) is calculated and returned as a 2x2 matrix.""",
    "video":"https://youtu.be/-BHegXJZAww",
    "learn": r'''
        ## Calculate Covariance Matrix

The covariance matrix is a fundamental concept in statistics, illustrating how much two random variables change together. It's essential for understanding the relationships between variables in a dataset.

For a dataset with n features, the covariance matrix is an n x n square matrix where each element (i, j) represents the covariance between the ith and jth features. Covariance is defined by the formula:

$$
\text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n-1}
$$

Where:

- X and Y are two random variables (features),
- x_i and y_i are individual observations of X and Y,
- x̄ (x-bar) and ȳ (y-bar) are the means of X and Y,
- n is the number of observations.

In the covariance matrix:

- The diagonal elements (where i = j) indicate the variance of each feature.
- The off-diagonal elements show the covariance between different features. This matrix is symmetric, as the covariance between X and Y is equal to the covariance between Y and X, denoted as cov(X, Y) = cov(Y, X).
''',
    "starter_code": "def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:\n    return covariance_matrix",
    "solution": """def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]

    means = [sum(feature) / n_observations for feature in vectors]

    for i in range(n_features):
        for j in range(i, n_features):
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_observations)) / (n_observations - 1)
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance

    return covariance_matrix""",
    "test_cases": [
        {
            "test": "calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]])",
            "expected_output": "[[1.0, 1.0], [1.0, 1.0]]"
        },
        {
            "test": "calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]])",
            "expected_output": "[[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]"
        }
    ]
}
,
    
"Singular Value Decomposition (SVD) (hard)": {
    "description": """Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by
      using the jacobian method and without using numpy svd function,
      i mean you could but you wouldn't learn anything. return the result i  this format.""",
    "example": """Example:
        input: a = [[2, 1], [1, 2]]
        output: (array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
        reasoning: U is the first matrix sigma is the second vector and V is the third matrix""",
    "learn": r'''
       ## Singular Value Decomposition (SVD) via the Jacobi Method

Singular Value Decomposition (SVD) is a powerful matrix decomposition technique in linear algebra that expresses a matrix as the product of three other matrices, revealing its intrinsic geometric and algebraic properties. When using the Jacobi method, SVD decomposes a matrix $$A$$ into:

$$
A = U\Sigma V^T
$$

- $$A$$ is the original $$m \times n$$ matrix.
- $$U$$ is an $$m \times m$$ orthogonal matrix whose columns are the left singular vectors of $$A$$.
- $$\Sigma$$ is an $$m \times n$$ diagonal matrix containing the singular values of $$A$$.
- $$V^T$$ is the transpose of an $$n \times n$$ orthogonal matrix whose columns are the right singular vectors of $$A$$.

### The Jacobi Method for SVD

The Jacobi method is an iterative algorithm used for diagonalizing a symmetric matrix through a series of rotational transformations. It is particularly suited for computing the SVD by iteratively applying rotations to minimize off-diagonal elements until the matrix is diagonal.

### Steps of the Jacobi SVD Algorithm

1. **Initialization**: Start with $$A^TA$$ (or $$AA^T$$ for $$U$$) and set $$V$$ (or $$U$$) as an identity matrix. The goal is to diagonalize $$A^TA$$, obtaining $$V$$ in the process.
   
2. **Choosing Rotation Targets**: Identify off-diagonal elements in $$A^TA$$ to be minimized or zeroed out through rotations.

3. **Calculating Rotation Angles**: For each target off-diagonal element, calculate the angle $$\theta$$ for the Jacobi rotation matrix $$J$$ that would zero it. This involves solving for $$\theta$$ using $$\text{atan2}$$ to accurately handle the quadrant of rotation:

   $$
   \theta = 0.5 \cdot \text{atan2}(2a_{ij}, a_{ii} - a_{jj})
   $$

   where $$a_{ij}$$ is the target off-diagonal element, and $$a_{ii}$$, $$a_{jj}$$ are the diagonal elements of $$A^TA$$.

4. **Applying Rotations**: Construct $$J$$ using $$\theta$$ and apply the rotation to $$A^TA$$, effectively reducing the magnitude of the target off-diagonal element. Update $$V$$ (or $$U$$) by multiplying it by $$J$$.

5. **Iteration and Convergence**: Repeat the process of selecting off-diagonal elements, calculating rotation angles, and applying rotations until $$A^TA$$ is sufficiently diagonalized.

6. **Extracting SVD Components**: Once diagonalized, the diagonal entries of $$A^TA$$ represent the squared singular values of $$A$$. The matrices $$U$$ and $$V$$ are constructed from the accumulated rotations, containing the left and right singular vectors of $$A$$, respectively.

### Practical Considerations

- The Jacobi method is particularly effective for dense matrices where off-diagonal elements are significant.
- Careful implementation is required to ensure numerical stability and efficiency, especially for large matrices.
- The iterative nature of the Jacobi method makes it computationally intensive, but it is highly parallelizable.

This iterative process underscores the adaptability of the Jacobi method for computing the SVD, offering a detailed insight into the matrix's structure through its singular values and vectors.
''',
    "starter_code": "import numpy as np \n svd_2x2_singular_values(A: np.ndarray) -> tuple: \n return SVD",
    "solution": """import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_T_A = A.T @ A
    theta = 0.5 * np.arctan2(2 * A_T_A[0, 1], A_T_A[0, 0] - A_T_A[1, 1])
    j = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    A_prime = j.T @ A_T_A @ j 
    
    # Calculate singular values from the diagonalized A^TA (approximation for 2x2 case)
    singular_values = np.sqrt(np.diag(A_prime))

    # Process for AA^T, if needed, similar to A^TA can be added here for completeness
    
    return j, singular_values, j.T""",
    "test_cases": [
        {
            "test": "svd_2x2_singular_values(np.array([[2, 1], [1, 2]]))",
            "expected_output": """(array([[ 0.70710678, -0.70710678],
       [ 0.70710678,  0.70710678]]), array([3., 1.]), array([[ 0.70710678,  0.70710678],
       [-0.70710678,  0.70710678]]))"""
        },
        {
            "test": "svd_2x2_singular_values(np.array([[1, 2], [3, 4]]))",
            "expected_output": """(array([[ 0.57604844, -0.81741556],
       [ 0.81741556,  0.57604844]]), array([5.4649857 , 0.36596619]), array([[ 0.57604844,  0.81741556],
       [-0.81741556,  0.57604844]]))"""
        }
    ]
}, 
"Determinant of a 4x4 Matrix using Laplace's Expansion (hard)": {
    "description": "Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.",
    "example": """Example:
        input: a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        output: 0
        reasoning: Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. Given the symmetrical and linear nature of this specific matrix, its determinant is 0. The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.""",
    "learn": r'''
        ## Determinant of a 4x4 Matrix using Laplace's Expansion

        Laplace's Expansion, also known as cofactor expansion, is a method to calculate the determinant of a square matrix of any size. For a 4x4 matrix \(A\), this method involves expanding \(A\) into minors and cofactors along a chosen row or column. 

        Consider a 4x4 matrix \(A\):
        $$
        A = \begin{pmatrix}
        a_{11} & a_{12} & a_{13} & a_{14} \\
        a_{21} & a_{22} & a_{23} & a_{24} \\
        a_{31} & a_{32} & a_{33} & a_{34} \\
        a_{41} & a_{42} & a_{43} & a_{44}
        \end{pmatrix}
        $$

        The determinant of \(A\), \(\det(A)\), can be calculated by selecting any row or column (e.g., the first row) and using the formula that involves the elements of that row (or column), their corresponding cofactors, and the determinants of the 3x3 minor matrices obtained by removing the row and column of each element. This process is recursive, as calculating the determinants of the 3x3 matrices involves further expansions.

        The expansion formula for the first row is as follows:
        $$
        \det(A) = a_{11}C_{11} - a_{12}C_{12} + a_{13}C_{13} - a_{14}C_{14}
        $$

        Here, \(C_{ij}\) represents the cofactor of element \(a_{ij}\), which is calculated as \((-1)^{i+j}\) times the determinant of the minor matrix obtained after removing the \(i\)th row and \(j\)th column from \(A\).
        ''',
    "starter_code": "def determinant_4x4(matrix: list[list[int|float]]) -> float:\n    # Your recursive implementation here\n    pass",
    "solution": """def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Base case: If the matrix is 1x1, return its single element
    if len(matrix) == 1:
        return matrix[0][0]
    # Recursive case: Calculate determinant using Laplace's Expansion
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]  # Remove column c
        cofactor = ((-1)**c) * determinant_4x4(minor)  # Compute cofactor
        det += matrix[0][c] * cofactor  # Add to running total
    return det""",
    "test_cases": [
        {"test": 'determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])', "expected_output": '0'},
        {"test": 'determinant_4x4([[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]])', "expected_output": '-160'},
        {"test": 'determinant_4x4([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])', "expected_output": '0'},
    ]
}}
