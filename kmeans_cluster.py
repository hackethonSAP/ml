# %%
import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df

# %%
from sklearn.cluster import KMeans

# Step 3: Define the number of clusters (k)
# We know there are 3 natural clusters in the Iris dataset
# Step 4: Initialize and fit the K-Means model
X = df[['sepal length (cm)', 'sepal width (cm)']]
# Step 4: Initialize and fit the K-Means model with k=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Predict cluster labels for the data
df['cluster'] = kmeans.labels_



# Simple scatter plot for visualization (Different colors for each cluster)
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=df['cluster'], cmap='viridis')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("K-Means Clustering (Iris Dataset)")
plt.show()
plt.show()


