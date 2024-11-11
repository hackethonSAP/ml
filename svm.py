# %%
from sklearn import datasets

data=datasets.load_iris()
data

# %%
x=data.data
y=data.target

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

model=SVC()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)




# %%
from sklearn.metrics import accuracy_score , precision_score

print(accuracy_score(y_pred,y_test))
print(precision_score(y_test,y_pred,average="macro",zero_division=1))

# Accuracy: The fraction of correctly classified samples.
# Precision: Precision is the ratio of correctly predicted positive observations to the total predicted positives. The average='macro' computes precision for each class and returns the unweighted mean.

# %%
#     Accuracy:
#         This is the proportion of correct predictions (correctly classified samples) out of all predictions. A perfect accuracy means that the classifier predicted all test samples correctly.

#     Precision:
#         This metric tells us how many of the predicted positive labels are actually positive. It is useful when the cost of false positives is high.
#         For multiclass classification (like Iris), we calculate the average precision across all classes. The macro average calculates precision for each class independently and then averages them.


# %%
# Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. 
# However, it is mostly used for classification problems. 
# The basic goal of SVM is to find a hyperplane that best separates the data into different classes.
#  It tries to maximize the margin between the classes.

# A hyperplane is a decision boundary that separates data points into different classes.
#  In 2D, it is a line; in 3D, it is a plane. 
# In higher dimensions, it's called a hyperplane.


