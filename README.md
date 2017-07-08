
## Titanic Survival Exploration
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

*Problem Reference* : https://www.kaggle.com/c/titanic

#### Dependencies
* numpy
* pandas
* matplotlib
* scikit-learn
* jupyter notebook


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
```


```python
# loading training dataset with labels
df = pd.read_csv("data/train.csv")
df.head(n=5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove the most incompleted fields 
df_reduced = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
df_clean = df_reduced.dropna()
df_clean.head(n=5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction import DictVectorizer
```


```python
# feature extraction from datasets
vec = DictVectorizer(sparse=False)
df_dict = df_clean.to_dict(orient = 'records')
df_dict_feature = vec.fit_transform(df_dict)
df_feature = vec.get_feature_names()
print df_feature
df_dict_feature
```

    ['Age', 'Embarked=C', 'Embarked=Q', 'Embarked=S', 'Fare', 'Parch', 'Pclass', 'Sex=female', 'Sex=male', 'SibSp', 'Survived']
    




    array([[ 22.,   0.,   0., ...,   1.,   1.,   0.],
           [ 38.,   1.,   0., ...,   0.,   1.,   1.],
           [ 26.,   0.,   0., ...,   0.,   0.,   1.],
           ..., 
           [ 19.,   0.,   0., ...,   0.,   0.,   1.],
           [ 26.,   1.,   0., ...,   1.,   0.,   1.],
           [ 32.,   0.,   1., ...,   1.,   0.,   0.]])




```python
df_final = pd.DataFrame(df_dict_feature,columns=df_feature)
X = df_final.drop('Survived',axis=1)
y = df_final['Survived']
df_final.head(n=5) #final prepared dataset with features and labels
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Embarked=C</th>
      <th>Embarked=Q</th>
      <th>Embarked=S</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex=female</th>
      <th>Sex=male</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#plotting of some insight of data male vs female
plt.figure(figsize=(12,2),dpi=80)
plt.subplot(1,2,1)
df_male=df_final[df_final['Sex=male']==1]
df_female=df_final[df_final['Sex=female']==1]
df_male['Survived'].value_counts().sort_index().plot(kind='barh',color='#1abc9c')
plt.xlabel('Number of male')
plt.ylabel('Survived(True/False)')
plt.title('Male Data of Survival')
plt.subplot(1,2,2)
df_female['Survived'].value_counts().sort_index().plot(kind='barh',color='#e74c3c')
plt.xlabel('Number of female')
plt.ylabel('Survived(True/False)')
plt.title('Female Data of Survival')
plt.show()
```


![png](resources/output_7_0.png)



```python
ratio_male= df_male['Survived'].sum()/len(df_male)
ratio_female = df_female['Survived'].sum()/len(df_female)

print 'Ratio of male survival',ratio_male
print 'Ratio of female survival',ratio_female
#As we see female have good chance of survival if you are female then kudos !! sorry boys
```

    Ratio of male survival 0.205298013245
    Ratio of female survival 0.752895752896
    


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42) 
```


```python
plt.figure(figsize=(6,4),dpi=100)
colors= ['#e74c3c','#2ecc71']
label_name = set(y_train)
plt.scatter(X_train.Age,X_train.Fare,c=y_train,alpha=0.7,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlim(-5,90)
plt.ylim(-20,300)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
```


![png](resources/output_10_0.png)



```python
#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf1 = RandomForestClassifier(n_estimators=15)
clf1.fit(X_train,y_train)
y_predict = clf1.predict(X_train)
print 'Model accuracy_score: ',accuracy_score(y_train,y_predict)
y_test_predict = clf1.predict(X_test)
print 'Test accuracy score',accuracy_score(y_test,y_test_predict)
```

    Model accuracy_score:  0.985324947589
    Test accuracy score 0.774468085106
    


```python
# Support vector machine for classification
from sklearn import svm
from sklearn.metrics import accuracy_score
clf2 = svm.SVC(gamma=0.00016,C=1000)
clf2.fit(X_train,y_train)
y_predict = clf2.predict(X_train)
print 'Model accuracy_score: ',accuracy_score(y_train,y_predict)
y_test_predict = clf2.predict(X_test)
print 'Test accuracy score',accuracy_score(y_test,y_test_predict)
```

    Model accuracy_score:  0.834381551363
    Test accuracy score 0.782978723404
    


```python
y_test_main_predict = clf2.predict(X_test)
```


```python
#final classifier with some good hyperparameter which we learn from tuning
clf3 = svm.SVC(gamma=0.00016,C=1000)
clf3.fit(X,y)
y_trainCSV_predict = clf3.predict(X)
print 'Model accuracy_score: ',accuracy_score(y,y_trainCSV_predict)
```

    Model accuracy_score:  0.824438202247
    


```python
# Cleaning test data
df_test = pd.read_csv("data/test_corrected.csv")
df_reduced_test = df_test.drop(['Unnamed: 11','Unnamed: 12','Unnamed: 13','PassengerId','Name','Ticket','Cabin','Unnamed: 0'],axis=1)
df_clean_test = df_reduced_test.dropna()
vec_test = DictVectorizer(sparse=False)
df_dict_test = df_clean_test.to_dict(orient = 'records')
df_dict_feature_test = vec_test.fit_transform(df_dict_test)
df_feature_test = vec_test.get_feature_names()
df_final_test = pd.DataFrame(df_dict_feature_test,columns=df_feature_test)
df_final_test.head(n=5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Embarked=C</th>
      <th>Embarked=Q</th>
      <th>Embarked=S</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex=female</th>
      <th>Sex=male</th>
      <th>SibSp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.8292</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.6875</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>8.6625</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12.2875</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#After clean test data and we see lots of age are missings .lets write model for predicting age
y_test_predict_final = clf3.predict(df_final_test)
df_test['Survived'] = y_test_predict_final
df_result = df_test[['PassengerId','Survived']]
df_result.to_csv('data/result.csv')
```

## Results:
The kaggle test data on this prediction model strategy gives accuracy of 76.56 % . 
