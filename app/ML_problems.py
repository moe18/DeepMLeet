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
    "Feature Scaling Implementation (easy)": {
        "description": "Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.",
        "example": """Example:
            input: data = np.array([[1, 2], [3, 4], [5, 6]])
            output: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            reasoning: Standardization rescales the feature to have a mean of 0 and a standard deviation of 1. Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value maps to 0 and the maximum to 1.""",
        "learn": r'''
            ## Feature Scaling Techniques

Feature scaling is crucial in many machine learning algorithms that are sensitive to the magnitude of features. This includes algorithms that use distance measures like k-nearest neighbors and gradient descent-based algorithms like linear regression.

### Standardization:
Standardization (or Z-score normalization) is the process where the features are rescaled so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one:
$$
z = \frac{(x - \mu)}{\sigma}
$$
Where \(x\) is the original feature, $$mu$$ is the mean of that feature, and $$sigma$$ is the standard deviation.

### Min-Max Normalization:
Min-max normalization rescales the feature to a fixed range, typically 0 to 1, or it can be shifted to any range \([a, b]\) by transforming the data according to the formula:
$$
x' = \frac{(x - \text{min}(x))}{(\text{max}(x) - \text{min}(x))} \times (\text{max} - \text{min}) + \text{min}
$$
Where $$x$$ is the original value, $${min}(x)$$ is the minimum value for that feature, $${max}(x)$$ is the maximum value, and $${min}$$ and $${max}$$ are the new minimum and maximum values for the scaled data.

Implementing these scaling techniques will ensure that the features contribute equally to the development of the model and improve the convergence speed of learning algorithms.
        ''',

        "starter_code": """def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    return standardized_data, normalized_data""",
        "solution": """
import numpy as np

def feature_scaling(data):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()""",
        "test_cases": [
            {
                "test": "feature_scaling(np.array([[1, 2], [3, 4], [5, 6]]))",
                "expected_output": "([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])"
            }
        ],
    }

    ,
    "K-Means Clustering(medium)": {
        "description": "Write a Python function that implements the k-Means algorithm for clustering, starting with specified initial centroids and a set number of iterations. The function should take a list of points (each represented as a tuple of coordinates), an integer k representing the number of clusters to form, a list of initial centroids (each a tuple of coordinates), and an integer representing the maximum number of iterations to perform. The function will iteratively assign each point to the nearest centroid and update the centroids based on the assignments until the centroids do not change significantly, or the maximum number of iterations is reached. The function should return a list of the final centroids of the clusters. Round to the nearest fourth decimal.",
        "example": """Example:
            input: points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
            output: [(1, 2), (10, 2)]
            reasoning: Given the initial centroids and a maximum of 10 iterations,
            the points are clustered around these points, and the centroids are
            updated to the mean of the assigned points, resulting in the final
            centroids which approximate the means of the two clusters.
            The exact number of iterations needed may vary,
            but the process will stop after 10 iterations at most.""",
        "learn": r'''
            ## Implementing k-Means Clustering

k-Means clustering is a method to partition `n` points into `k` clusters. Here is a brief overview of how to implement the k-Means algorithm:

1. **Initialization**: Start by selecting `k` initial centroids. These can be randomly selected from the dataset or based on prior knowledge.

2. **Assignment Step**: For each point in the dataset, find the nearest centroid. The "nearest" can be defined using Euclidean distance. Assign the point to the cluster represented by this nearest centroid.

3. **Update Step**: Once all points are assigned to clusters, update the centroids by calculating the mean of all points in each cluster. This becomes the new centroid of the cluster.

4. **Iteration**: Repeat the assignment and update steps until the centroids no longer change significantly, or until a predetermined number of iterations have been completed. This iterative process helps in refining the clusters to minimize within-cluster variance.

5. **Result**: The final centroids represent the center of the clusters, and the points are partitioned accordingly.

This algorithm assumes that the `mean` is a meaningful measure, which might not be the case for non-numeric data. The choice of initial centroids can significantly affect the final clusters, hence multiple runs with different starting points can lead to a more comprehensive understanding of the cluster structure in the data.
        ''',

        "starter_code": """def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    # Your code here
    return final_centroids""",
        "solution": """
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]""",
        "test_cases": [
            {
                "test": "k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10)",
                "expected_output": "[(1.0, 2.0), (10.0, 2.0)]"
            },
            {
                "test": "k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10)",
                "expected_output": "[(1.0, 1.0, 1.0), (10.3333, 10.6667, 10.3333)]"
            }
        ],
    },
    "Cross-Validation Data Split Implementation (medium)": {
        "description": "Write a Python function that performs k-fold cross-validation data splitting from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature) and an integer k representing the number of folds. The function should split the dataset into k parts, systematically use one part as the test set and the remaining as the training set, and return a list where each element is a tuple containing the training set and test set for each fold.",
        "example": """Example:
            input: data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k = 5
            output: [[[[3, 4], [5, 6], [7, 8], [9, 10]], [[1, 2]]],
                    [[[1, 2], [5, 6], [7, 8], [9, 10]], [[3, 4]]],
                    [[[1, 2], [3, 4], [7, 8], [9, 10]], [[5, 6]]], 
                    [[[1, 2], [3, 4], [5, 6], [9, 10]], [[7, 8]]], 
                    [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10]]]]
            reasoning: The dataset is divided into 5 parts, each being used once as a test set while the remaining parts serve as the training set.""",
        "learn": r'''
            ## Understanding k-Fold Cross-Validation Data Splitting

k-Fold cross-validation is a technique used to evaluate the generalizability of a model by dividing the data into `k` folds or subsets. Each fold acts as a test set once, with the remaining `k-1` folds serving as the training set. This approach ensures that every data point gets used for both training and testing, improving model validation.

### Steps in k-Fold Cross-Validation Data Split:

1. **Shuffle the dataset randomly**. (but not in this case becuase we test for a unique result)
2. **Split the dataset into k groups**.
3. **Generate Data Splits**: For each group, treat that group as the test set and the remaining groups as the training set.

### Benefits of this Approach:

- Ensures all data is used for both training and testing.
- Reduces bias since each data point gets to be in a test set exactly once.
- Provides a more robust estimate of model performance.

Implementing this data split function will allow a deeper understanding of how data partitioning affects machine learning models and will provide a foundation for more complex validation techniques.
        ''',

        "starter_code": """def cross_validation_split(data: np.ndarray, k: int) -> list:
    # Your code here
    return folds""",
        "solution": """
import numpy as np

def cross_validation_split(data, k):
    np.random.shuffle(data)  # This line can be removed if shuffling is not desired in examples
    fold_size = len(data) // k
    folds = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size if i != k-1 else len(data)
        test = data[start:end]
        train = np.concatenate([data[:start], data[end:]])
        folds.append([train.tolist(), test.tolist()])
    
    return folds""",
        "test_cases": [
            {
                "test": "cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 2)",
                "expected_output": 
                    """[[[[5, 6], [7, 8], [9, 10]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]]]"""
            }
        ],
    }

,


    "Principal Component Analysis (PCA) Implementation (medium)": {
        "description": "Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.",
        "example": """Example:
            input: data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
            output:  [[0.7071], [0.7071]]
            reasoning: After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.""",
        "learn": r'''
            ## Understanding Eigenvalues in PCA

Principal Component Analysis (PCA) utilizes the concept of eigenvalues and eigenvectors to identify the principal components of a dataset. Here's how eigenvalues fit into the PCA process:

### Eigenvalues and Eigenvectors: The Foundation of PCA

For a given square matrix \(A\), representing the covariance matrix in PCA, eigenvalues \(\lambda\) and their corresponding eigenvectors \(v\) satisfy:
$$
Av = \lambda v
$$

### Calculating Eigenvalues

The eigenvalues of matrix \(A\) are found by solving the characteristic equation:
$$
\det(A - \lambda I) = 0
$$
where \(I\) is the identity matrix of the same dimension as \(A\). This equation highlights the relationship between a matrix, its eigenvalues, and eigenvectors.

### Role in PCA

In PCA, the covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.

### Eigenvalues and Dimensionality Reduction

The magnitude of an eigenvalue correlates with the importance of its corresponding eigenvector (principal component) in representing the dataset's variability. By selecting a subset of eigenvectors corresponding to the largest eigenvalues, PCA achieves dimensionality reduction while preserving as much of the dataset's variability as possible.

### Practical Application

1. **Standardize the Dataset**: Ensure that each feature has a mean of 0 and a standard deviation of 1.
2. **Compute the Covariance Matrix**: Reflects how features vary together.
3. **Find Eigenvalues and Eigenvectors**: Solve the characteristic equation for the covariance matrix.
4. **Select Principal Components**: Choose eigenvectors (components) with the highest eigenvalues for dimensionality reduction.

Through this process, PCA transforms the original features into a new set of uncorrelated features (principal components), ordered by the amount of original variance they explain.
        ''',

        "starter_code": """import numpy as np \ndef pca(data: np.ndarray, k: int) -> list[list[int|float]]:
    # Your code here
    return principal_components""",
        "solution": """
import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4).tolist()""",
        "test_cases": [
            {
                "test": "pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2)",
                "expected_output": "[[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]"
            },
            {
                "test": "pca(np.array([[1, 2], [3, 4], [5, 6]]), k = 1)",
                "expected_output": " [[0.7071], [0.7071]]"
            }
        ],
    }

    ,
    
  "Decision Tree Learning (hard)": {
    "description": "Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.",
    "example":"""
Example:
    input: examples = [
                {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
            ],
            attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    output: {
        'Outlook': {
            'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
            'Overcast': 'Yes',
            'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
        }
    }
    reasoning: Using the given examples, the decision tree algorithm determines that 'Outlook' is the best attribute to split the data initially. When 'Outlook' is 'Overcast', the outcome is always 'Yes', so it becomes a leaf node. In cases of 'Sunny' and 'Rain', it further splits based on 'Humidity' and 'Wind', respectively. The resulting tree structure is able to classify the training examples with the attributes 'Outlook', 'Temperature', 'Humidity', and 'Wind'.
""",
        "learn": "## Decision Tree Learning Algorithm\n\nThe decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.\n\nThe recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, which are calculated as follows:\n\nEntropy: $$H(X) = -\\sum p(x) \\log_2 p(x)$$ \n\n Information Gain: $$IG(D, A) = H(D) - \\sum \\frac{|D_v|}{|D|} H(D_v)$$\n\nWhere:\n- $$H(X)$$ is the entropy of the set,\n- $$IG(D, A)$$ is the information gain of dataset $$D$$ after splitting on attribute $$A$$,\n- $$D_v$$ is the subset of $$D$$ for which attribute $$A$$ has value $$v$$.\n\nThe attribute with the highest information gain is used at each step, and the dataset is split based on this attribute's values. This process continues recursively until all data is perfectly classified or no remaining attributes can be used to make a split.",
    "starter_code": "def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:\n    # Your code here\n    return decision_tree",
    "solution": """
import math
from collections import Counter

def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())
    return entropy

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    gains = {attr: calculate_information_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    
    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree
""",
    "test_cases": [
    {
        "test": "learn_decision_tree([\n"
                "    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},\n"
                "    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},\n"
                "    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                "    {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},\n"
                "    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                "    {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n"
                "    {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},\n"
                "    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}\n"
                "], ['Outlook', 'Wind'], 'PlayTennis')",
        "expected_output":  "{'Outlook': {'Sunny': {'Wind': {'Weak': 'No', 'Strong': 'No'}}, 'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}, 'Overcast': 'Yes'}}"
    }
]


  }}

