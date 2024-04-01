problems = {
    "1. Machine Learning Fundamentals": {
        "section": True,
        "description": """## Machine Learning in Data Analysis and Prediction

Machine Learning (ML) is a pivotal technology for data analysis, prediction, and automated decision-making. This section delves into the core concepts and algorithms of Machine Learning, providing a pathway to mastering the techniques that enable computers to learn from and make predictions based on data:

### Foundational Skills

- **Supervised vs Unsupervised Learning**: Understand the differences between these two main types of learning and when to use each.
- **Data Preprocessing**: Learn the importance of data cleaning, normalization, and splitting datasets for training and testing.
- **Feature Engineering and Selection**: Discover how to enhance model performance by selecting or creating new features from existing data.
- **Model Evaluation**: Grasp the concepts of overfitting, underfitting, and how to use metrics to evaluate a model's performance.

### Intermediate Understanding

- **Classification Algorithms**: Dive into algorithms like Decision Trees, Support Vector Machines (SVM), and Random Forests, understanding their workings and applications.
- **Regression Analysis**: Explore linear and non-linear regression models to predict continuous outcomes.
- **Clustering Techniques**: Learn about K-Means, Hierarchical Clustering, and DBSCAN for segmenting datasets into meaningful groups.

### Advanced Techniques

- **Deep Learning**: Introduce the fundamentals of neural networks, including convolutional and recurrent neural networks, for complex problems like image and speech recognition.
- **Ensemble Methods**: Understand how combining models through techniques like Bagging and Boosting can improve prediction accuracy.
- **Dimensionality Reduction**: Learn about Principal Component Analysis (PCA) and t-SNE for reducing the dimensionality of data while preserving its structure.

Each topic is crafted to build upon the previous, ensuring a deep and comprehensive understanding of Machine Learning's role in data analysis and predictive modeling. Practical exercises and examples will illustrate how to apply these concepts to real-world scenarios, preparing learners to develop and evaluate their own machine learning models.""",
        "example": "",
        "learn": "",
        "starter_code": "",
        "solution": "",
        "test_cases": []
    }
,
    "Entropy Calculation (easy)": {
        "description": "Write a Python function that calculates the entropy of a dataset. The function should take a list of class labels as input and return the entropy of the dataset. Round your answer to four decimal places.",
        "example": """Example:
            input: labels = [1, 1, 1, 0, 0, 0]
            output: 1.0
            reasoning: There are two classes (1 and 0), each occurring 3 times out of 6, leading to a probability of 0.5 for each class. The entropy is -2 * (0.5 * log2(0.5)) = 1.0.""",
        "learn": r'''
            ## Entropy in Information Theory and Machine Learning

Entropy is a measure of the unpredictability or randomness of a dataset. In the context of machine learning, particularly in decision tree algorithms, entropy can be used to quantify the impurity or disorder within a set of items. It helps in determining the best features that contribute to dividing the dataset into the best possible homogeneous sets.

The formula for entropy of a dataset is:

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \cdot \log_2(p(x_i))
$$

Where:
- $$H(X)$$ is the entropy of the set $$X$$,
- $$p(x_i)$$ is the probability of occurrence of class $$x_i$$ in the set,
- The sum is calculated over all unique classes in the dataset.

Entropy is 0 when the dataset is completely uniform (all elements belong to a single class), and it reaches its maximum value when the dataset is evenly split between classes.

### Practical Implementation

Calculating the entropy of a dataset involves determining the frequency of each class in the dataset, calculating the probability of each class, and then applying the entropy formula.
        ''',

        "starter_code": """import numpy as np
def calculate_entropy(labels: list) -> float:
    # Your code here, make sure to round
    return entropy""",
        "solution": """
import numpy as np
def calculate_entropy(labels: list) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return np.round(entropy, 4)""",
        "test_cases": [
            {
                "test": "calculate_entropy([1, 1, 1, 0, 0, 0])",
                "expected_output": "1.0"
            },
            {
                "test": "calculate_entropy([1, 1, 1, 1, 0, 0, 0, 0])",
                "expected_output": "1.0"
            },
            {
                "test": "calculate_entropy([1, 1, 0, 0, 0, 0])",
                "expected_output": "0.9183"
            },
            {
                "test": "calculate_entropy([1, 1, 1, 1, 1, 0])",
                "expected_output": "0.65"
            }
        ],
    },

    "Linear Regression Using Normal Equation (easy)": {
    "description": "Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.",
    "example": """Example:
        input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
        output: [0.0, 1.0]
        reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.""",
    "learn": r'''
        ## Linear Regression Using the Normal Equation

Linear regression aims to model the relationship between a scalar dependent variable $$y$$ and one or more explanatory variables (or independent variables) $$X$$. The normal equation provides an analytical solution to finding the coefficients $$\theta$$ that minimize the cost function for linear regression.

Given a matrix $$X$$ (with each row representing a training example and each column a feature) and a vector $$y$$ (representing the target values), the normal equation is:

$$
\theta = (X^TX)^{-1}X^Ty
$$

Where:
- $$X^T$$ is the transpose of $$X$$,
- $$(X^TX)^{-1}$$ is the inverse of the matrix $$X^TX$$,
- $$y$$ is the vector of target values.

**Things to note**: This method does not require any feature scaling, and there's no need to choose a learning rate. However, computing the inverse of $$X^TX$$ can be computationally expensive if the number of features is very large.

### Practical Implementation

A practical implementation involves augmenting $$X$$ with a column of ones to account for the intercept term and then applying the normal equation directly to compute $$\theta$$.
    ''',

    "starter_code": """import numpy as np\n def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    return theta""",
    "solution": """
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    theta = np.round(theta,4).flatten().tolist()
    return theta""",
    "test_cases": [
        {
            "test": "linear_regression_normal_equation([[1,1], [1,2], [1,3]], [1, 2, 3])",
            "expected_output": "[-0.0, 1.0]"
        },
        {
            "test": "linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1,2,1])",
            "expected_output": "[4.0, -1.0, -0.0]"
        }
    ],
},
    "Linear Regression Using Gradient Descent (easy)": {
        "description": "Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.",
        "example": """Example:
            input: X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
            output: np.array([0.1107, 0.9513])
            reasoning: The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.""",
        "learn": r'''
            ## Linear Regression Using Gradient Descent

Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:

$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_j^{(i)}
$$

Where:
- $$\alpha$$ is the learning rate,
- $$m$$ is the number of training examples,
- $$h_{\theta}(x^{(i)})$$ is the hypothesis function at iteration $$i$$,
- $$x^{(i)}$$ is the feature vector of the $$i$$th training example,
- $$y^{(i)}$$ is the actual target value for the $$i$$th training example,
- $$x_j^{(i)}$$ is the value of feature $$j$$ for the $$i$$th training example.

**Things to note**: The choice of learning rate and the number of iterations are crucial for the convergence and performance of gradient descent. Too small a learning rate may lead to slow convergence, while too large a learning rate may cause overshooting and divergence.

### Practical Implementation

Implementing gradient descent involves initializing the weights, computing the gradient of the cost function, and iteratively updating the weights according to the update rule.
        ''',

        "starter_code": """import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1))
    return theta""",
        "solution": """
import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y.reshape(-1, 1)
        updates = X.T @ errors / m
        theta -= alpha * updates
    return np.round(theta.flatten(), 4)""",
        "test_cases": [
            {
                "test": "linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000)",
                "expected_output": "[0.1107 0.9513]"
            },
            {
                "test": "linear_regression_gradient_descent(np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]]), np.array([2, 3, 5]), 0.1, 10)",
                "expected_output": "[-1.0241 -1.9133 -3.9616]"
            }
        ],
    },

}