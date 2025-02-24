### k-Nearest Neighbors (k-NN) – Key Points  

1. **Definition**:  
   k-NN is a **supervised learning** algorithm used for **classification** and **regression** tasks. It predicts the class/label of a data point based on the majority class of its nearest neighbors.

2. **Working Principle**:  
   - Select a value for **k** (number of neighbors).  
   - Measure the **distance** (Euclidean, Manhattan, Minkowski, etc.) between the query point and all training points.  
   - Identify the **k nearest neighbors**.  
   - Assign the class label based on the **majority vote** (for classification) or **average value** (for regression).  

3. **Distance Metrics**:  
   - **Euclidean Distance**: $( d(p, q) = \sqrt{\sum (p_i - q_i)^2} $) (most common)  
   - **Manhattan Distance**: $( d(p, q) = \sum |p_i - q_i| $)  
   - **Minkowski Distance**: Generalized form of Euclidean & Manhattan  

4. **Choosing k**:  
   - **Small k** → Sensitive to noise, may lead to overfitting.  
   - **Large k** → More generalization but may ignore local patterns.  
   - **Common choice**: k = √(n) (where n = number of data points)  

5. **Pros & Cons**:  
   ✅ Simple to implement, No training phase  
   ✅ Works well with low-dimensional, small datasets  
   ❌ Computationally expensive for large datasets (O(n) for each query)  
   ❌ Sensitive to irrelevant features & scale differences (requires normalization)  

6. **Optimization Techniques**:  
   - **Feature scaling** (Standardization/Normalization)  
   - **Dimensionality reduction** (PCA, LDA)  
   - **KD-Trees / Ball Trees** (for faster nearest neighbor search)  

7. **Use Cases**:  
   - Image classification  
   - Recommender systems  
   - Anomaly detection  
   - Medical diagnosis  
