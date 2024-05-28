problems = {
    "1. NumPy": {
        "section": True,
        "description": """ ## Numpy in Machine Learning and Deep Learning
NumPy is the foundation for numerical computations in Machine Learning and Deep Learning. It efficiently stores and manipulates multidimensional data (matrices, vectors) used in algorithms like linear regression,
 neural networks, and image processing. Its speed and functionality for linear algebra operations make it crucial for building and training Machine Learning and Deep Learning models.
### Foundational Operations

- **Get Array Shape:** Knowing the dimensions (shape) of a NumPy array is essential for data manipulation and model building. You can use the `.shape` attribute to access the shape information.
- **Array Creation and Manipulation:** Efficiently create NumPy arrays from various data structures like lists using `np.array()` and perform element-wise access and modification using indexing (e.g., `arr[0]`) and slicing.
- **Mathematical Operations:** Apply arithmetic operations directly on arrays for vectorized calculations. NumPy's broadcasting mechanism allows for compatible operations between arrays of different shapes.

### Intermediate Understanding

- **Matrix Multiplication and Linear Transformations:** Utilize NumPy functions like `np.dot(arr1, arr2)` for matrix multiplication and other linear algebra operations like dot product. These are fundamental for neural networks and understanding how data is transformed in machine learning.
- **Element-wise Array Operations:** Perform element-wise mathematical operations on arrays. This is crucial for feature scaling (e.g., normalization) commonly used in machine learning pipelines.

### Advanced Techniques

- **Random Number Generation:** Generate random numbers from various distributions (normal, uniform) using functions like `np.random.rand(n)`, `np.random.normal(mean, std)`. This is useful for simulations and data augmentation techniques in machine learning.
- **Statistical Functions:** Calculate summary statistics (mean, median, standard deviation) on NumPy arrays with functions like `np.mean(arr)`, `np.median(arr)`, `np.std(arr)`. This helps understand data characteristics and prepare it for machine learning models.
- **Filtering and Indexing:** Select specific elements based on conditions or boolean masks using square brackets `[arr > 5]`. This allows for efficient data analysis and manipulation tasks.
""",
        "example": "",
        "learn": "",
        "starter_code": "",
        "solution": "",
        "test_cases": [],
    },
    "Create a NumPy Array (easy)": {
        "description": "Write a Python function that creates a NumPy array from a given list.",
        "example": """Example:
    input: [1, 2, 3, 4, 5]
    output: [1 2 3 4 5]
    reasoning: The given list is converted into a NumPy array.""",
        "video": "Coming Soon",
        "learn": r"""
    ## Creating a NumPy Array

    NumPy arrays are the foundation of nearly all numerical computation in Python. They are similar to Python lists but offer more features, such as being faster and requiring less memory. Here's how you can create a NumPy array:

    ```python
    import numpy as np

    # Create a NumPy array from a list
    my_list = [1, 2, 3, 4, 5]
    my_array = np.array(my_list)
    ```

    This will create a NumPy array `my_array` from the list `my_list`.
    """,
        "starter_code": "import numpy as np\n\ndef create_numpy_array(lst: list[int|float]) -> np.ndarray:\n    # Write your code here\n    return",
        "solution": """import numpy as np

def create_numpy_array(lst: list[int|float]) -> np.ndarray:
    return np.array(lst)""",
        "test_cases": [
            {"test": "create_numpy_array([1, 2, 3, 4, 5])", "expected_output": "[1 2 3 4 5]"},
            {"test": "create_numpy_array([3.1, 2.7, 1.6])", "expected_output": "[3.1 2.7 1.6]"},
            {"test": "create_numpy_array([-1, -2, -3, -4, -5])", "expected_output": "[-1 -2 -3 -4 -5]"},
        ],
    },
}
