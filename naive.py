# %%
import pandas as pd

data={
    'feature1' : [1,2,3,4,5],
    'feature2' : [5,4,3,2,1],
    'target'   : [1,2,3,4,5]
}

df=pd.DataFrame(data)
df

# %%
from sklearn.model_selection import train_test_split

x=df[['feature1','feature2']]
y=df['target']


# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# %%
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

# %%
from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))


