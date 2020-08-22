#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import scipy.stats as stats
#IMPORT DATASET
data=pd.read_csv('HumanBreastCancer_data_set.csv')
data.rename(columns = {"concave points_mean": "concave_points_mean","concave points_se": "concave_points_se","concave points_worst": "concave_points_worst"},inplace=True )

data
data.info()
data.head()
data['diagnosis'].unique()
X=data.drop(['id','diagnosis'],axis=1)
X
#ENCODING CATEGORICAL INTO BINARY
def enc(a):
    if a=='M':
        return 1
    else:
        return 0
y=data['diagnosis'].apply(enc)
#COUNT PLOT
M=0
B=0
for i in y:
    if(i==1):
        M=M+1;
    else:
        B=B+1;
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
ax = sns.countplot(y,label="Count")     
ax.set_title("NO OF BENIGN AND MALIGNANT IN THE DATA SET")
y.head()
X.head()
X.describe()
#DATA VISUALISATION
x_mean=list(data.columns[1:12])
plt.figure(figsize=(30,30))
sns.heatmap(data[x_mean].corr(),annot=True,square=True,cmap="coolwarm",linewidth=0.6)
plt.show()
x_se=list(data.columns[12:22])
plt.figure(figsize=(30,30))
sns.heatmap(data[x_se].corr(),annot=True,square=True,cmap="coolwarm",linewidth=0.6)
plt.show()

x_worst=list(data.columns[22:32])
plt.figure(figsize=(30,30))
sns.heatmap(data[x_worst].corr(),annot=True,square=True,cmap="coolwarm",linewidth=0.6)
plt.show()
col=X.columns
col
#CORRELATION
for i in col:
    print(np.corrcoef(X[i],y))
#DROP COLUMNS
X.drop(columns=["texture_se","symmetry_se","fractal_dimension_mean","smoothness_se","fractal_dimension_se","perimeter_mean","area_mean","perimeter_se","perimeter_worst","area_worst","area_se","concave_points_mean","Unnamed: 32"],inplace=True)
X.columns
#BOX PLOT
data1=pd.DataFrame(np.random.rand(10,31),columns=col)
data1.plot.box(grid="True")
#SPLITTING INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#SCALING
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)
#LOGISTIC REGRESSION:
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
from sklearn.metrics import roc_curve,roc_auc_score
roc_auc_score(y_test,y_pred1)
fpr,tpr,thresholds=roc_curve(y_test,y_pred1)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')
plt.show()
# SUPPORTED VECTOR MACHINE:
from sklearn.svm import SVC
model2=SVC()
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
print(classification_report(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred2)
from sklearn.metrics import roc_curve,roc_auc_score
roc_auc_score(y_test,y_pred2)
fpr,tpr,thresholds=roc_curve(y_test,y_pred2)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')
plt.show()
# DECISION TREE:
from sklearn.tree import DecisionTreeClassifier
model3=DecisionTreeClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)
print(classification_report(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred3)
from sklearn.metrics import roc_curve,roc_auc_score
roc_auc_score(y_test,y_pred3)
fpr,tpr,thresholds=roc_curve(y_test,y_pred3)
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')
plt.show()
