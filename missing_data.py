# %%
import pandas as pd

data=pd.read_csv('region.csv')
data

# %%
data.isnull()

# %%
#data['Income'].fillna(0,inplace=True)
#missing values 
data['Income'].fillna(data['Income'].mean(),inplace=True)
data

# %%
#yes or not like 1 or 0
data_region=pd.get_dummies(data,columns=['Region'])
data_region

# %%
from sklearn.model_selection import train_test_split

help(train_test_split)

# %%


x=data[['Age','Income']]
y=data['Online Shopper']

x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=2)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaled = scaler.fit_transform(data[['Income']])

scaled




