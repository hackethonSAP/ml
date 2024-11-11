# %%
import numpy as np

data={
    'feature1':[2,3,5,11],
    'feature2':[8,9,10,12],
    'feature3':[3,4,11,12]
}


# %%
import pandas as pd

df=pd.DataFrame(data)



# %%
from sklearn.preprocessing import StandardScaler

# Step 1: Standardize the data (important for PCA)
scaler =StandardScaler() 
df_scaler=scaler.fit_transform(df) # Scaling to mean=0 and std=1

# %%
# Why standardize? PCA is affected by the scale of the data. Features with larger scales will dominate the analysis. By scaling, we give each feature equal weight.
from sklearn.decomposition import PCA

pca=PCA()
df_pca=pca.fit_transform(df_scaler)

df_pca=pd.DataFrame(df_pca)

# Step 3: Convert the result back to a DataFrame
print(df_pca)

#Principal Component Analysis (PCA) is a technique used for dimensionality reduction. 
#It transforms data from a high-dimensional space into a lower-dimensional space while preserving as much information (variance) as possible. 
#In other words, PCA helps you reduce the number of features in your dataset while retaining the most important aspects of the data.

#Standardize the Data: PCA is affected by the scale of the data, so it's common to standardize or normalize the data before applying PCA


