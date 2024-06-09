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
        "test_cases": [],
        "use_micro": False
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
            "test": "print(sigmoid(0))",
            "expected_output": "0.5"
        },
        {
            "test": "print(sigmoid(1))",
            "expected_output": "0.7311"
        },
        {
            "test": "print(sigmoid(-1))",
            "expected_output": "0.2689"
        }
    ],
    "use_micro": False
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
            "test": "print(softmax([1, 2, 3]))",
            "expected_output": "[0.09, 0.2447, 0.6652]"
        },
        {
            "test": "print(softmax([1, 1, 1]))",
            "expected_output": "[0.3333, 0.3333, 0.3333]"
        },
        {
            "test": "print(softmax([-1, 0, 5]))",
            "expected_output": "[0.0025, 0.0067, 0.9909]"
        }
    ],
    "use_micro": False
},
"Single Neuron (easy)": {
    "description": "Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.",
    "example": """Example:
        input: features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
        output: ([0.4626, 0.4134, 0.6682], 0.3349)
        reasoning: For each input vector, the weighted sum is calculated by multiplying each feature by its corresponding weight, adding these up along with the bias, then applying the sigmoid function to produce a probability. The MSE is calculated as the average squared difference between each predicted probability and the corresponding true label.""",
    "learn": r'''
        ## Single Neuron Model with Multidimensional Input and Sigmoid Activation

This task involves a neuron model designed for binary classification with multidimensional input features, using the sigmoid activation function to output probabilities. It also involves calculating the mean squared error (MSE) to evaluate prediction accuracy.

### Mathematical Background:

1. **Neuron Output Calculation**:
   $$ z = \sum (weight_i \times feature_i) + bias $$

   $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

2. **MSE Calculation**:
   $$ MSE = \frac{1}{n} \sum (predicted - true)^2 $$

Where:
- $$z$$ is the sum of weighted inputs plus bias,
- $$\sigma(z)$$ is the sigmoid activation output,
- $$predicted$$ are the probabilities after sigmoid activation,
- $$true$$ are the true binary labels.

### Practical Implementation:

- Each feature vector is processed to calculate a combined weighted sum, which is then passed through the sigmoid function to determine the probability of the input belonging to the positive class.
- MSE provides a measure of error, offering insights into the model's performance and aiding in its optimization.
    ''',

    "starter_code": """import math\n\ndef single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):\n    # Your code here\n    return probabilities, mse""",
    "solution": """
import math
def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse""",
    "test_cases": [
        {
            "test": "print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))",
            "expected_output": "([0.4626, 0.4134, 0.6682], 0.3349)"
        },
        {
            "test": "print(single_neuron_model([[1, 2], [2, 3], [3, 1]], [1, 0, 1], [0.5, -0.2], 0))",
            "expected_output": " ([0.525, 0.5987, 0.7858], 0.21)"
        }
    ],
},
"Single Neuron with Backpropagation (medium)": {
    "description": "Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.",
    "example": """Example:
        input: features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
        output: updated_weights = [0.0808, -0.1916], updated_bias = -0.0214, mse_values = [0.2386, 0.2348]
        reasoning: The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.""",
    "learn": r'''
        ## Neural Network Learning with Backpropagation

This question involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

### Mathematical Background:

1. **Forward Pass**:
   - Compute the neuron output by calculating the dot product of the weights and input features and adding the bias:

     $$ z = w_1x_1 + w_2x_2 + ... + w_nx_n + b $$

   - Apply the sigmoid activation function to convert the linear combination into a probability:

     $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

2. **Loss Calculation (MSE)**:
   - The Mean Squared Error is used to quantify the error between the neuron's predictions and the actual labels:

     $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2 $$

3. **Backward Pass (Gradient Calculation)**:
   - Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
     
     $$ \frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij} $$

     $$ \frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) $$

4. **Parameter Update**:
   - Update each weight and the bias by subtracting a portion of the gradient determined by the learning rate:

     $$ w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j} $$

     $$ b = b - \alpha \frac{\partial MSE}{\partial b} $$

### Practical Implementation:

This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.
    ''',

    "starter_code": """import numpy as np\n\ndef train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):\n    # Your code here\n    return updated_weights, updated_bias, mse_values""",
    "solution": """
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients / len(labels)
        bias -= learning_rate * bias_gradient / len(labels)

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values""",
    "test_cases": [
        {
            "test": "print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))",
            "expected_output": "([0.1019, -0.1711], -0.0083, [0.3033, 0.2987])"
        },
        {
            "test": "print(train_neuron(np.array([[1, 2], [2, 3], [3, 1]]), np.array([1, 0, 1]), np.array([0.5, -0.2]), 0, 0.1, 3))",
            "expected_output": "([0.4943, -0.2155], 0.0013, [0.21, 0.2093, 0.2087])"
        }
    ],
    "use_micro": False
},


"Implementing Basic Autograd Operations (medium)": {
    "description": "Special thanks to Andrej Karpathy for making a video about this, if you havent already check out his videos on youtube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.",
    "example": """Example:
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
        Output: Value(data=2, grad=0) Value(data=-3, grad=10) Value(data=10, grad=-3) Value(data=-28, grad=1) Value(data=0, grad=1)
        Explanation: The output reflects the forward computation and gradients after backpropagation. The ReLU on 'd' zeros out its output and gradient due to the negative data value.""",
    "learn": r'''
    ## Understanding Mathematical Concepts in Autograd Operations
First off watch this: https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg

This task focuses on the implementation of basic automatic differentiation mechanisms for neural networks. The operations of addition, multiplication, and ReLU are fundamental to neural network computations and their training through backpropagation.

### Mathematical Foundations:

1. **Addition (`__add__`)**:
   - **Forward pass**: For two scalar values $$ a $$ and $$b $$, their sum \( s \) is simply \( s = a + b \).
   - **Backward pass**: The derivative of \( s \) with respect to both \( a \) and \( b \) is 1. Therefore, during backpropagation, the gradient of the output is passed directly to both inputs.

2. **Multiplication (`__mul__`)**:
   - **Forward pass**: For two scalar values \( a \) and \( b \), their product \( p \) is \( p = a \times b \).
   - **Backward pass**: The gradient of \( p \) with respect to \( a \) is \( b \), and with respect to \( b \) is \( a \). This means that during backpropagation, each input's gradient is the product of the other input and the output's gradient.

3. **ReLU Activation (`relu`)**:
   - **Forward pass**: The ReLU function is defined as \( R(x) = \max(0, x) \). This function outputs \( x \) if \( x \) is positive and 0 otherwise.
   - **Backward pass**: The derivative of the ReLU function is 1 for \( x > 0 \) and 0 for \( x \leq 0 \). Thus, the gradient is propagated through the function only if the input is positive; otherwise, it stops.

### Conceptual Application in Neural Networks:

- **Addition and Multiplication**: These operations are ubiquitous in neural networks, forming the basis of computing weighted sums of inputs in the neurons.
- **ReLU Activation**: Commonly used as an activation function in neural networks due to its simplicity and effectiveness in introducing non-linearity, making learning complex patterns possible.

Understanding these operations and their implications on gradient flow is crucial for designing and training effective neural network models. By implementing these from scratch, one gains deeper insights into the workings of more sophisticated deep learning libraries.
''',

    "starter_code": """class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Implement addition here
        pass

    def __mul__(self, other):
        # Implement multiplication here
        pass

    def relu(self):
        # Implement ReLU here
        pass

    def backward(self):
        # Implement backward pass here
        pass""",
    "solution": """
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
""",
    "test_cases": [
        {
            "test": """a = Value(2);b = Value(3);c = Value(10);d = a + b * c  ;e = Value(7) * Value(2);f = e + d;g = f.relu()  
g.backward()
print(a,b,c,d,e,f,g)
""",
            "expected_output": """ Value(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)"""
        }
    ],
    "use_micro": False
},
  "Implementing and Training a Simple Neural Network (medium)": {
    "description": "Inspired by the foundational work in neural networks, your task is to build a basic multi-layer perceptron (MLP) that can be trained on a small dataset using stochastic gradient descent (SGD). The network should include forward and backward propagation capabilities for training. Implement a network with one hidden layer and ReLU activation, followed by an output layer with a linear activation for regression tasks. The network should also compute the mean squared error (MSE) loss and perform parameter updates via backpropagation.",
    "example": """Example:
      # Network setup
      mlp = MLP(2, [3, 1])  # 2 input features, 3 neurons in hidden layer, 1 output
      # Training data (simple xor function)
      inputs = [Value(0.5), Value(0.5)]
      target = Value(0.25)
      # Training loop
      for epoch in range(100):  # Train for 100 epochs
        output = mlp(inputs)
        loss = (output - target) ** 2  # MSE loss
        mlp.zero_grad()  # Zero gradients before backpropagation
        loss.backward()  # Compute gradients
        mlp.update_parameters(0.01)  # Update parameters with learning rate of 0.01
        print(f'Epoch {epoch}, Loss: {loss.data}')
      Output: Display loss per epoch to monitor training progress.
      Explanation: This setup trains the MLP on fixed data to minimize MSE loss, demonstrating basic network training dynamics.""",
    "learn": r'''
      ## Building and Training a Neural Network
      This task involves constructing a neural network that includes all necessary components for supervised learning, including:
      
      ### Components:
      - **Neuron**: Fundamental unit that computes weighted inputs plus a bias.
      - **Layer**: Composes multiple neurons to transform inputs to outputs.
      - **MLP**: Integrates layers to form a complete network.
      - **Forward Pass**: Computes the network's output.
      - **Loss Calculation**: Measures the network's prediction error.
      - **Backward Pass**: Applies the chain rule to compute gradients for learning.
      - **Parameter Update**: Adjusts the network's weights based on gradients to reduce loss.

      ### Conceptual Workflow:
      - Data is processed by the network to produce predictions.
      - Loss is calculated from the predictions and true data.
      - Gradients are calculated to understand how to adjust weights to improve predictions.
      - Parameters are updated based on these gradients to minimize future loss.

      By understanding each step, you gain insights into how neural networks learn from data, which is crucial for developing more complex AI systems.
    ''',
    "starter_code": """class MLP(Module):
        # Define your MLP architecture here
        # Implement forward, backward, and parameter update methods

    def train_network():
        mlp = MLP(...)
        # Define your training loop

    train_network()
    """,
    "solution": """
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i<len(nouts)-1) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)[0]  # Assuming single output from final layer for simplicity
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def update_parameters(self, lr):
        for p in self.parameters():
            p.data -= lr * p.grad

# Example usage
def train_network():
    mlp = MLP(2, [3, 1])
    inputs = [Value(0.5), Value(-0.1)]
    target = Value(0.4)

    for epoch in range(100):
        output = mlp(inputs)
        loss = (output - target) ** 2
        mlp.zero_grad()
        loss.backward()
        mlp.update_parameters(0.01)
        print(f'Epoch {epoch}, Loss: {loss.data}')

train_network()
    """,
    "test_cases": [
        {
            "test": "",
            "expected_output": "Display of training loss per epoch to monitor progress"
        }
    ],
    'use_micro':True
}
}




