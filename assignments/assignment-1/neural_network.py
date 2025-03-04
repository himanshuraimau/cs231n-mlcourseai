import numpy as np
import matplotlib.pyplot as plt
from data_utils import generate_spiral_data

def train_neural_network(X, y, hidden_size=100, learning_rate=1e-0, reg=1e-3, num_iters=10000):
    """
    Train a two-layer neural network for classification
    Args:
        X: input data (N, D)
        y: labels (N,)
        hidden_size: number of neurons in hidden layer
        learning_rate: step size for gradient descent
        reg: regularization strength
        num_iters: number of iterations for optimization
    Returns:
        W, b: weights and biases for first layer
        W2, b2: weights and biases for second layer
    """
    num_examples = X.shape[0]  # Number of training examples
    input_size = X.shape[1]    # Input dimension
    num_classes = np.max(y) + 1  # Number of classes
    
    # Initialize neural network parameters
    W = 0.01 * np.random.randn(input_size, hidden_size)  # First layer weights
    b = np.zeros((1, hidden_size))                       # First layer bias
    W2 = 0.01 * np.random.randn(hidden_size, num_classes)  # Second layer weights
    b2 = np.zeros((1, num_classes))                        # Second layer bias
    
    # Training loop
    for i in range(num_iters):
        # Forward pass
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # ReLU activation
        scores = np.dot(hidden_layer, W2) + b2          # Output layer scores
        
        # Compute softmax probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Compute loss = data loss + regularization loss
        correct_logprobs = -np.log(probs[range(num_examples),y])  # Cross-entropy loss
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*(np.sum(W*W) + np.sum(W2*W2))  # L2 regularization
        loss = data_loss + reg_loss
        
        if i % 1000 == 0:
            print(f"iteration {i}: loss {loss}")
        
        # Backward pass
        # Compute gradients
        dscores = probs  # Initial upstream gradient
        dscores[range(num_examples),y] -= 1  # Subtract 1 from correct class scores
        dscores /= num_examples
        
        # Backpropagation: output layer -> hidden layer
        dW2 = np.dot(hidden_layer.T, dscores)  # Gradient for W2
        db2 = np.sum(dscores, axis=0, keepdims=True)  # Gradient for b2
        
        # Backpropagation: hidden layer -> input layer
        dhidden = np.dot(dscores, W2.T)  # Gradient at hidden layer
        dhidden[hidden_layer <= 0] = 0  # ReLU gradient: zero out negative inputs
        
        # Input layer gradients
        dW = np.dot(X.T, dhidden)  # Gradient for W
        db = np.sum(dhidden, axis=0, keepdims=True)  # Gradient for b
        
        # Add regularization gradients
        dW2 += reg * W2
        dW += reg * W
        
        # Parameter updates using gradient descent
        W += -learning_rate * dW   # Update first layer weights
        b += -learning_rate * db   # Update first layer bias
        W2 += -learning_rate * dW2 # Update second layer weights
        b2 += -learning_rate * db2 # Update second layer bias
    
    return W, b, W2, b2

if __name__ == "__main__":
    # Generate training data
    X, y = generate_spiral_data()
    
    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
    # Train the network
    W, b, W2, b2 = train_neural_network(X, y)
    
    # Evaluate training accuracy
    hidden_layer = np.maximum(0, np.dot(X, W) + b)  # Forward pass through first layer
    scores = np.dot(hidden_layer, W2) + b2          # Forward pass through second layer
    predicted_class = np.argmax(scores, axis=1)     # Get predicted classes
    accuracy = np.mean(predicted_class == y)        # Calculate accuracy
    print(f'Training accuracy: {accuracy:.2f}')
