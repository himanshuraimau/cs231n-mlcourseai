Softmax Classifier – Key Points
Definition:
The Softmax Classifier is a supervised learning algorithm used for multi-class classification. It is an extension of logistic regression that assigns probabilities to each class.

Softmax Function:
Given a vector of raw scores (logits) $z$ for each class, the softmax function converts them into probabilities:

$P(y_i = j \mid x_i) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$
 
$z_j$ is the raw score (logit) for class $j$.
$K$ is the total number of classes.
Each probability is between 0 and 1, and the sum of all probabilities is 1.
Loss Function – Cross-Entropy Loss:
The softmax classifier uses cross-entropy loss, which measures the difference between the predicted probability distribution and the actual class labels:

$L = -\sum_{i=1}^{N} \sum_{j=1}^{K} y_{ij} \log P(y_i = j \mid x_i)$
$y_{ij}$ is 1 if the actual class of $x_i$ is $j$, otherwise 0.
The loss is minimized when the predicted probability for the correct class is maximized.
Gradient Descent Optimization:

The weights $W$ are updated using gradient descent:
$W \leftarrow W - \eta \nabla L$
where $\eta$ is the learning rate.
The gradient of the loss w.r.t. $W$ is:
$\frac{\partial L}{\partial W} = X^T (P - Y)$
where:
$P$ is the predicted probability matrix.
$Y$ is the one-hot encoded true label matrix.
Pros & Cons:
✅ Works well for multi-class classification
✅ Outputs interpretable probabilities
❌ Sensitive to outliers
❌ Assumes classes are mutually exclusive

Use Cases:

Image classification (e.g., digit recognition)
Sentiment analysis
NLP tasks