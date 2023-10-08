import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Iris.csv')
df.head()

df.shape

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.duplicated().sum()

df.shape

df['Species'].value_counts()

x = df.iloc[:,:-1]
# x = df.drop('label',axis=1)
# x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df.iloc[:,-1]
# y = df['Species']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


sns.scatterplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'],hue=df['Species'])
plt.show()

sns.scatterplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'],hue=df['Species'])
plt.show()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

m1 = KNeighborsClassifier(n_neighbors=11)
m1.fit(x_train,y_train)

# Accuracy
print('Training score',m1.score(x_train,y_train))
print('Testing score',m1.score(x_test,y_test))


ypred = m1.predict(x_test)
print(ypred)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,ypred)
print(cm)
print(classification_report(y_test,ypred))


x_train.head()


sns.scatterplot(x=df['SepalLengthCm'],y=df['SepalWidthCm'],hue=df['Species'])
plt.scatter([4.8,5.3],[3.3,2.5],color='black',marker='*',s=120)
plt.show()