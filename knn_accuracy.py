# %%
import pandas as pd

data = {
    "feature1": [1, 2, 3, 6, 7, 8],  # Feature 1 values
    "feature2": [5, 6, 7, 2, 3, 4],  # Feature 2 values
    "target": [0, 0, 0, 1, 1, 1]  # Target labels (binary classification)
}

df=pd.DataFrame(data)
df

# %%
from sklearn.model_selection import train_test_split

x=df[['feature1','feature2']]
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

# %%
from sklearn.neighbors import KNeighborsClassifier

# Step 4: Initialize the KNN model with k=3 (3 nearest neighbors)
model=KNeighborsClassifier(n_neighbors=1)

# The fit() function trains the model on the training data.
model.fit(x_train,y_train)




# %%
from sklearn.metrics import accuracy_score

y_pred=model.predict(x_test)

print(accuracy_score(y_test,y_pred))

# %%
# The K-Nearest Neighbors (KNN) model is a simple, instance-based learning algorithm used for classification (and sometimes regression). It works by finding the k closest data points (neighbors) to a new data point and classifying it based on the majority class among those neighbors.

#     Core Idea: KNN assumes that similar points are close in distance. When a new data point needs classification, it is assigned the label most common among its k nearest neighbors.
#     Distance Metric: Typically, Euclidean distance is used to measure "closeness," but other distance metrics (like Manhattan) can also be applied.
#     Parameter k: The choice of k (number of neighbors) affects the model’s performance. Smaller values of k make the model sensitive to noise, while larger values make it more generalized.


