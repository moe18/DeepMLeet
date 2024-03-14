problems = {
    "Add Two Numbers": {
        "description": "Write a Python function that accepts two parameters and returns their sum.",
        "starter_code": "def add(a, b):\n    return a + b",
        "solution": "def add(a, b):\n    return a + b  # This is the simplest solution.",
        "test_cases": [
            {"test": "add(1, 2)", "expected_output": "3"},
            {"test": "add(-1, -1)", "expected_output": "-2"},
            {"test": "add(100, 200)", "expected_output": "300"},
        ],
    },
    "Linear Regression with NumPy": {
        "description": "Implement linear regression using only NumPy to fit a model to the given data points. The function should accept two parameters: X (features) and y (target) and return the coefficients of the linear model.",
        "starter_code": """import numpy as np

def linear_regression(X, y):
    # Add a bias term with ones
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Calculate coefficients using the Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best
""",
        "solution": """import numpy as np

def linear_regression(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best  # Returns the vector of coefficients [b, m1, m2, ..., mn]

# Example usage (not part of the solution code):
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
# theta_best = linear_regression(X, y)
""",
        "test_cases": [
            {
                "test": "linear_regression(np.array([[1], [2], [3]]), np.array([3, 5, 7]))",
                "expected_output": "[1. 2.]",
            },
            {
                "test": "linear_regression(np.array([[1, 2], [2, 3], [3, 4]]), np.array([6, 8, 10]))",
                "expected_output": "[2. 1. 1.]",
            },
            {
                "test": "linear_regression(np.array([[1], [2], [4]]), np.array([2, 3, 6]))",
                "expected_output": "[0.66666667 1.33333333]",
            },
        ],
    },
    # Include other problems and their solutions in the same manner
}
