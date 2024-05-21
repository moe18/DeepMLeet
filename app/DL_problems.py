problems = {
    "Deep Learning": {
        "section": True,
        "description": """## Deep Learning

Deep Learning (DL) is a transformative branch of machine learning, pivotal for handling complex data analysis and prediction tasks. This section explores the essential concepts and architectures of Deep Learning, equipping you with the knowledge to harness the power of neural networks:

### Foundational Skills

- **Neural Network Basics**: Understand the architecture of neural networks, including neurons, layers, and activation functions.
- **Backpropagation and Training**: Learn how neural networks learn through the backpropagation algorithm and the role of loss functions and optimizers.
- **Data Preprocessing for DL**: Discover the specific requirements for preparing data for deep learning, such as normalization and data augmentation.

### Intermediate Understanding

- **Convolutional Neural Networks (CNNs)**: Dive into CNNs for image processing, including layer types like convolutional layers, pooling layers, and fully connected layers.
- **Recurrent Neural Networks (RNNs)**: Explore RNNs for sequence data analysis, useful in applications like language modeling and time series prediction.
- **Regularization Techniques**: Understand methods like dropout and batch normalization to prevent overfitting in deep neural networks.

### Advanced Techniques

- **Advanced Architectures**: Learn about advanced neural network architectures such as Transformers and GANs (Generative Adversarial Networks) for tasks like natural language processing and generative models.
- **Transfer Learning**: Grasp how pre-trained models can be adapted for new tasks, significantly reducing the need for large labeled datasets.
- **Autoencoders**: Explore how autoencoders can be used for unsupervised learning tasks, including dimensionality reduction and feature learning.

Each topic is designed to build on the previous, ensuring a thorough understanding of Deep Learning's role in modern data analysis and predictive modeling. Practical exercises and real-world examples will demonstrate how to apply these concepts effectively, preparing learners to develop and evaluate their own deep learning models.""",        "example": "",
        "learn": "",
        "starter_code": "",
        "solution": "",
        "test_cases": []
    },
    "Sigmoid Activation Function Understanding (easy)": {
    "description": "Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.",
    "example": """Example:
        input: z = 0
        output: 0.5
        reasoning: The sigmoid function is defined as σ(z) = 1 / (1 + exp(-z)). For z = 0, exp(-0) = 1, hence the output is 1 / (1 + 1) = 0.5.""",
    "learn": r'''
        ## Understanding the Sigmoid Activation Function

The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the (0, 1) interval, making it useful for modeling probability as an output.

### Mathematical Definition:

The sigmoid function is mathematically defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where \(z\) is the input to the function.

### Characteristics:

- **Output Range**: The output is always between 0 and 1.
- **Shape**: It has an "S" shaped curve.
- **Gradient**: The function's gradient is highest near z = 0 and decreases toward either end of the z-axis.

This function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.
    ''',

    "starter_code": """import math\n\ndef sigmoid(z: float) -> float:\n    # Your code here\n    return result""",
    "solution": """
import math
def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)""",
    "test_cases": [
        {
            "test": "sigmoid(0)",
            "expected_output": "0.5"
        },
        {
            "test": "sigmoid(1)",
            "expected_output": "0.7311"
        },
        {
            "test": "sigmoid(-1)",
            "expected_output": "0.2689"
        }
    ],
},
"Softmax Activation Function Implementation (easy)": {
    "description": "Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.",
    "example": """Example:
        input: scores = [1, 2, 3]
        output: [0.0900, 0.2447, 0.6652]
        reasoning: The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.""",
    "learn": r'''
        ## Understanding the Softmax Activation Function

The softmax function is a generalization of the sigmoid function and is used in the output layer of a neural network model that handles multi-class classification tasks.

### Mathematical Definition:

The softmax function is mathematically represented as:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

Where:
- \(z_i\) is the score for class \(i\),
- The denominator is the sum of the exponentials of all the scores.

### Characteristics:

- **Output Range**: Each output value is between 0 and 1, and the sum of all outputs is 1.
- **Purpose**: It transforms scores into probabilities, which are easier to interpret and are useful for classification.

This function is essential for models where the output needs to represent a probability distribution across multiple classes.
    ''',

    "starter_code": """import math\n\ndef softmax(scores: list[float]) -> list[float]:\n    # Your code here\n    return probabilities""",
    "solution": """
import math
def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    return probabilities""",
    "test_cases": [
        {
            "test": "softmax([1, 2, 3])",
            "expected_output": "[0.09, 0.2447, 0.6652]"
        },
        {
            "test": "softmax([1, 1, 1])",
            "expected_output": "[0.3333, 0.3333, 0.3333]"
        },
        {
            "test": "softmax([-1, 0, 5])",
            "expected_output": "[0.0025, 0.0067, 0.9909]"
        }
    ],
}



}




