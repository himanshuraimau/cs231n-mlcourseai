import numpy as np
import matplotlib.pyplot as plt
from data_utils import generate_spiral_data

def train_linear_classifier(X, y, learning_rate=1e-0, reg=1e-3, num_iters=200):
    """
    Train a linear classifier using softmax loss
    Args:
        X: input data (N, D)
        y: labels (N,)
        learning_rate: step size for gradient descent
        reg: regularization strength
        num_iters: number of iterations for optimization
    Returns:
        W: trained weights
        b: trained bias
    """
    num_examples = X.shape[0]  # Number of training examples
    num_features = X.shape[1]  # Input dimension
    num_classes = np.max(y) + 1  # Number of classes (assumes y starts from 0)
    
    # Initialize parameters with small random values to break symmetry
    W = 0.01 * np.random.randn(num_features, num_classes)  # Weight matrix
    b = np.zeros((1, num_classes))  # Bias vector
    
    # Gradient descent loop
    for i in range(num_iters):
        # Forward pass
        scores = np.dot(X, W) + b  # Linear scoring: XW + b
        exp_scores = np.exp(scores)  # Exponentiate scores for softmax
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax probabilities
        
        # Compute loss = data loss + regularization loss
        correct_logprobs = -np.log(probs[range(num_examples), y])  # Cross-entropy loss
        data_loss = np.sum(correct_logprobs)/num_examples  # Average loss per example
        reg_loss = 0.5*reg*np.sum(W*W)  # L2 regularization
        loss = data_loss + reg_loss  # Total loss
        
        if i % 10 == 0:
            print(f"iteration {i}: loss {loss}")
        
        # Compute gradients
        dscores = probs  # Initial upstream gradient is softmax probabilities
        dscores[range(num_examples), y] -= 1  # Subtract 1 from correct class scores
        dscores /= num_examples  # Normalize by batch size
        
        # Backpropagation
        dW = np.dot(X.T, dscores)  # Gradient for weights
        db = np.sum(dscores, axis=0, keepdims=True)  # Gradient for bias
        dW += reg*W  # Add regularization gradient
        
        # Parameter update using gradient descent
        W += -learning_rate * dW  # Update weights
        b += -learning_rate * db  # Update bias
    
    return W, b

if __name__ == "__main__":
    # Generate training data
    X, y = generate_spiral_data()
    
    # Visualize the generated data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
    # Train the classifier
    W, b = train_linear_classifier(X, y)
    
    # Evaluate training accuracy
    scores = np.dot(X, W) + b  # Compute final scores
    predicted_class = np.argmax(scores, axis=1)  # Get predicted classes
    accuracy = np.mean(predicted_class == y)  # Calculate accuracy
    print(f'Training accuracy: {accuracy:.2f}')
