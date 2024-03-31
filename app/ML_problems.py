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
    "Linear Regression Using Normal Equation (medium)": {
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
}