### **Support Vector Machine (SVM) – Key Points**  

1. **Definition**:  
   SVM is a **supervised learning algorithm** used for **classification and regression**. It finds the best **hyperplane** that maximally separates data points from different classes.  

2. **Key Concept – Hyperplane & Margin**:  
   - A **hyperplane** is a decision boundary that separates different classes.  
   - **Support Vectors** are the data points closest to the hyperplane, which influence its position.  
   - **Margin** is the distance between the hyperplane and the nearest support vectors. **SVM maximizes this margin** for better generalization.  

3. **Mathematical Formulation**:  
   Given a dataset with **feature vectors** \(x_i\) and class labels \(y_i\), where \(y_i \in \{-1, 1\}\), the **SVM optimization problem** is:  

   **Hard Margin SVM (for linearly separable data)**  
   $$
   \min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2
   $$
   Subject to:  
   $$
   y_i (\mathbf{w} \cdot x_i + b) \geq 1, \quad \forall i
   $$  

   **Soft Margin SVM (for non-linearly separable data)** (adds slack variable \(\xi_i\))  
   $$
   \min_{\mathbf{w}, b, \xi} \frac{1}{2} ||\mathbf{w}||^2 + C \sum \xi_i
   $$
   Subject to:  
   $$
   y_i (\mathbf{w} \cdot x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
   $$  
   - \( C \) is a hyperparameter that controls the trade-off between **margin size** and **misclassification penalty**.  

4. **Kernel Trick (For Nonlinear Data)**:  
   SVM can transform data into higher dimensions using a **kernel function** to make it linearly separable. Common kernels:  
   - **Linear Kernel**: \( K(x_i, x_j) = x_i \cdot x_j \)  
   - **Polynomial Kernel**: $( K(x_i, x_j$) = $(x_i $ $cdot x_j $ + c)^d $)  
   - **Radial Basis Function (RBF) Kernel**:  
     $$
     K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)
     $$  
   - **Sigmoid Kernel**: $( K(x_i, x_j$) = $tanh($alpha $x_i$ cdot $ x_j$ + c))  

5. **Pros & Cons**:  
   ✅ Works well for high-dimensional data  
   ✅ Effective in complex, non-linear problems (with kernel trick)  
   ❌ Computationally expensive for large datasets  
   ❌ Sensitive to hyperparameter tuning (C, kernel, gamma)  

6. **Use Cases**:  
   - **Text classification** (spam detection, sentiment analysis)  
   - **Image recognition** (face detection)  
   - **Anomaly detection**  
