
<h1>Human Activity
Recognition Using Smartphones</h1>

<h2>Data Wrangling and Preprocessing</h2>

<h3>Import all required Modules</h3>


```
#Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from matplotlib import cm
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import re
from math import *
```

<h3>File Paths</h3>


```
feature_path = '/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT4/AssessmentTask4/features.txt'
X_train_path = '/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT4/AssessmentTask4/train/X_train.txt'
y_train_path = '/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT4/AssessmentTask4/train/y_train.txt'
X_test_path = '/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT4/AssessmentTask4/test/X_test.txt'
y_test_path = '/Users/ragyibrahim/Documents/Uni/2019/Trimester 2/SIT720/AT4/AssessmentTask4/test/y_test.txt'
```

<h3>Clean String Function</h3>


```
#Clean Text
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_text(text):
    text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
    text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]
    
    return text
```

<h3>Import Training and Testing Data function</h3>


```
def import_X(df_path, feature_path):
    df = pd.read_csv(df_path, lineterminator='\n', header= None, names="a")
    feature_names = pd.read_csv(feature_path, sep = '\t', header=None, names= "F")
    preprocess_text(feature_names['F'])
    df = df.a.str.split(expand=True)
    df.columns = feature_names['F']
    df = df.apply(lambda x: pd.to_numeric(x))
    if 'train' in df_path:
        globals()['Xtrain'] = df
    else:
        globals()['Xtest'] = df
    return df.head()
```


```
#Import Training data
import_X(X_train_path, feature_path)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>F</th>
      <th>1 tBodyAcc-mean()-X</th>
      <th>2 tBodyAcc-mean()-Y</th>
      <th>3 tBodyAcc-mean()-Z</th>
      <th>4 tBodyAcc-std()-X</th>
      <th>5 tBodyAcc-std()-Y</th>
      <th>6 tBodyAcc-std()-Z</th>
      <th>7 tBodyAcc-mad()-X</th>
      <th>8 tBodyAcc-mad()-Y</th>
      <th>9 tBodyAcc-mad()-Z</th>
      <th>10 tBodyAcc-max()-X</th>
      <th>...</th>
      <th>552 fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>553 fBodyBodyGyroJerkMag-skewness()</th>
      <th>554 fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>555 angle(tBodyAccMean,gravity)</th>
      <th>556 angle(tBodyAccJerkMean),gravityMean)</th>
      <th>557 angle(tBodyGyroMean,gravityMean)</th>
      <th>558 angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>559 angle(X,gravityMean)</th>
      <th>560 angle(Y,gravityMean)</th>
      <th>561 angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.074323</td>
      <td>-0.298676</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>0.158075</td>
      <td>-0.595051</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.414503</td>
      <td>-0.390748</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.404573</td>
      <td>-0.117290</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>0.087753</td>
      <td>-0.351471</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>




```
#Import Testing data
import_X(X_test_path, feature_path)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>F</th>
      <th>1 tBodyAcc-mean()-X</th>
      <th>2 tBodyAcc-mean()-Y</th>
      <th>3 tBodyAcc-mean()-Z</th>
      <th>4 tBodyAcc-std()-X</th>
      <th>5 tBodyAcc-std()-Y</th>
      <th>6 tBodyAcc-std()-Z</th>
      <th>7 tBodyAcc-mad()-X</th>
      <th>8 tBodyAcc-mad()-Y</th>
      <th>9 tBodyAcc-mad()-Z</th>
      <th>10 tBodyAcc-max()-X</th>
      <th>...</th>
      <th>552 fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>553 fBodyBodyGyroJerkMag-skewness()</th>
      <th>554 fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>555 angle(tBodyAccMean,gravity)</th>
      <th>556 angle(tBodyAccJerkMean),gravityMean)</th>
      <th>557 angle(tBodyGyroMean,gravityMean)</th>
      <th>558 angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>559 angle(X,gravityMean)</th>
      <th>560 angle(Y,gravityMean)</th>
      <th>561 angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.257178</td>
      <td>-0.023285</td>
      <td>-0.014654</td>
      <td>-0.938404</td>
      <td>-0.920091</td>
      <td>-0.667683</td>
      <td>-0.952501</td>
      <td>-0.925249</td>
      <td>-0.674302</td>
      <td>-0.894088</td>
      <td>...</td>
      <td>0.071645</td>
      <td>-0.330370</td>
      <td>-0.705974</td>
      <td>0.006462</td>
      <td>0.162920</td>
      <td>-0.825886</td>
      <td>0.271151</td>
      <td>-0.720009</td>
      <td>0.276801</td>
      <td>-0.057978</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.286027</td>
      <td>-0.013163</td>
      <td>-0.119083</td>
      <td>-0.975415</td>
      <td>-0.967458</td>
      <td>-0.944958</td>
      <td>-0.986799</td>
      <td>-0.968401</td>
      <td>-0.945823</td>
      <td>-0.894088</td>
      <td>...</td>
      <td>-0.401189</td>
      <td>-0.121845</td>
      <td>-0.594944</td>
      <td>-0.083495</td>
      <td>0.017500</td>
      <td>-0.434375</td>
      <td>0.920593</td>
      <td>-0.698091</td>
      <td>0.281343</td>
      <td>-0.083898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.275485</td>
      <td>-0.026050</td>
      <td>-0.118152</td>
      <td>-0.993819</td>
      <td>-0.969926</td>
      <td>-0.962748</td>
      <td>-0.994403</td>
      <td>-0.970735</td>
      <td>-0.963483</td>
      <td>-0.939260</td>
      <td>...</td>
      <td>0.062891</td>
      <td>-0.190422</td>
      <td>-0.640736</td>
      <td>-0.034956</td>
      <td>0.202302</td>
      <td>0.064103</td>
      <td>0.145068</td>
      <td>-0.702771</td>
      <td>0.280083</td>
      <td>-0.079346</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.270298</td>
      <td>-0.032614</td>
      <td>-0.117520</td>
      <td>-0.994743</td>
      <td>-0.973268</td>
      <td>-0.967091</td>
      <td>-0.995274</td>
      <td>-0.974471</td>
      <td>-0.968897</td>
      <td>-0.938610</td>
      <td>...</td>
      <td>0.116695</td>
      <td>-0.344418</td>
      <td>-0.736124</td>
      <td>-0.017067</td>
      <td>0.154438</td>
      <td>0.340134</td>
      <td>0.296407</td>
      <td>-0.698954</td>
      <td>0.284114</td>
      <td>-0.077108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.274833</td>
      <td>-0.027848</td>
      <td>-0.129527</td>
      <td>-0.993852</td>
      <td>-0.967445</td>
      <td>-0.978295</td>
      <td>-0.994111</td>
      <td>-0.965953</td>
      <td>-0.977346</td>
      <td>-0.938610</td>
      <td>...</td>
      <td>-0.121711</td>
      <td>-0.534685</td>
      <td>-0.846595</td>
      <td>-0.002223</td>
      <td>-0.040046</td>
      <td>0.736715</td>
      <td>-0.118545</td>
      <td>-0.692245</td>
      <td>0.290722</td>
      <td>-0.073857</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>




```
#Print dataset features and rows
print("The training dataset has {} rows and {} features".format(Xtrain.shape[0], Xtrain.shape[1]))
print("The testing dataset has {} rows and {} features".format(Xtest.shape[0], Xtest.shape[1]))
```

    The training dataset has 7352 rows and 561 features
    The testing dataset has 2947 rows and 561 features


<h3>Import Training and Testing Response Variables</h3>


```
#Import Y-Train data
yTrain = pd.read_csv(y_train_path, header= None, names='Y')
```


```
#Import Y-Train data
yTest = pd.read_csv(y_test_path, header= None, names='Y')
```


```
#Map Unique Numbers to Class Names
def map_classes(df):
    mydict={
        1:'Walking',
            2:'Walking_Upstairs',
            3:'Walking_Downstairs', 
            4:'Sitting', 
            5:'Standing', 
            6:'Laying'
           }
    i = 0
    for item in df:
        if(i>=0 and item in mydict):
            continue
        else:    
           i = i+1
           mydict[item] = i+1

    k=[]
    for item in df:
        k.append(mydict[item])
    return k
```


```
#Add Class Names to Training outcome variable
yTrain = pd.DataFrame(map_classes(yTrain['Y']))
yTrain.columns = ['Y']
yTrain_label = yTrain['Y']
```


```
#Add Class Names to Testing outcome variable
yTest = pd.DataFrame(map_classes(yTest['Y']))
yTest.columns = ['Y']
yTest_label = yTest['Y']
```


```
#Get unique classes
uniqueClasses = sorted(yTrain['Y'].unique().tolist(), reverse=False)
```


```
#Function to print and plot class distribution
def freqViz(df):
    g = df.iloc[:,0].value_counts()
    h = df.iloc[:,0].value_counts().plot('bar')
    plt.show()
    return g , h
```


```
freqViz(yTest)
```


![png](output_21_0.png)





    (Laying                537
     Standing              532
     Walking               496
     Sitting               491
     Walking_Upstairs      471
     Walking_Downstairs    420
     Name: Y, dtype: int64,
     <matplotlib.axes._subplots.AxesSubplot at 0x1a20f7db50>)



From the above graph we can see that there's a clear class imbalance - therefore accuracy is not a good measure of model performace as it rewards models that predict the most prevelant class. Also, using 


```
#Find missing values
Xtrain[Xtrain.isnull().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>F</th>
      <th>1 tBodyAcc-mean()-X</th>
      <th>2 tBodyAcc-mean()-Y</th>
      <th>3 tBodyAcc-mean()-Z</th>
      <th>4 tBodyAcc-std()-X</th>
      <th>5 tBodyAcc-std()-Y</th>
      <th>6 tBodyAcc-std()-Z</th>
      <th>7 tBodyAcc-mad()-X</th>
      <th>8 tBodyAcc-mad()-Y</th>
      <th>9 tBodyAcc-mad()-Z</th>
      <th>10 tBodyAcc-max()-X</th>
      <th>...</th>
      <th>552 fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>553 fBodyBodyGyroJerkMag-skewness()</th>
      <th>554 fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>555 angle(tBodyAccMean,gravity)</th>
      <th>556 angle(tBodyAccJerkMean),gravityMean)</th>
      <th>557 angle(tBodyGyroMean,gravityMean)</th>
      <th>558 angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>559 angle(X,gravityMean)</th>
      <th>560 angle(Y,gravityMean)</th>
      <th>561 angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 561 columns</p>
</div>



<h3>Dimensionality Reduction</h3>

<h4>Principle Component Analysis</h4>


```
#Principle Component Analysis
pca = PCA(n_components=100)
pca.fit(Xtrain)

#Find the variance captured by each Principle Axis
pca_variance = pca.explained_variance_ratio_

#Plot Cumilative Variance
pca_cum = np.cumsum(np.round(pca_variance, decimals=4)*100)
plt.figure()
plt.plot(pca_cum, '-')
plt.xlabel("Number of Principle Axis")
plt.ylabel("Cumilative Variance Captured")
plt.grid(which='both')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward= True)
```


![png](output_25_0.png)


From the above graph we can see that at K = 65 (i.e. 65 principle compenents) w can capture ~95% of the variation included in the data. This will allows us to retain as much information gain as possible while dramatically decreaing the number of varibles we use in our models (from 560 to 65)

<h4>Fit PCA model to data</h4>


```
#Fit PCA model and apply reduction
pca_final = PCA(n_components=65)
Xtrain_pca = pca_final.fit_transform(Xtrain)
Xtest_pca = pca_final.fit_transform(Xtest)
#Examine new shape
print(Xtrain_pca.shape)
print(Xtest_pca.shape)
```

    (7352, 65)
    (2947, 65)


<h2>Machine Learning</h2>

<h3>Using F1-Scoring in GridsearchCV</h3>


```
#Using F1-Scoring
f1 = make_scorer(f1_score , average='macro')
```


```
#Cross Validation with Shuffling
from sklearn.model_selection import KFold
cv_shuffle = KFold(n_splits = 10,
                   shuffle = True,
                   random_state = 42)
```

<h3>K-Nearest Neighbour Classifier</h3>


```
#Set Range for K
k_range = range(1,51)
knn_param_grid = { 
    'n_neighbors': k_range
}
```


```
#build KNN using 5 neighbours
knn_class = KNeighborsClassifier()
```

<h4>Cross-Validation</h4>


```
knn_cv = GridSearchCV(estimator = knn_class,
                      param_grid =knn_param_grid,
                      cv=cv_shuffle,
                      scoring = f1,
                      return_train_score = True)
```


```
knn_cv_fit = knn_cv.fit(Xtrain_pca, 
                        yTrain_label)
```


```
#print best parameters 
print ("The optimal K-Nearest Neighbour Model has {} neighbours".format(knn_cv_fit.best_params_['n_neighbors']))
```

    The optimal K-Nearest Neighbour Model has 3 neighbours


<h4>Fit Optimal Model</h4>


```
#Build Optimal KNN Classifier
knn_class_optimal = KNeighborsClassifier(n_neighbors= 3)
```


```
#Fit model to training data
knn_class_optimal.fit(Xtrain_pca, yTrain_label)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=3, p=2,
               weights='uniform')




```
#Predict Using final model
knn_optimal_predict = knn_class_optimal.predict(Xtest_pca)
```

<h4>Evaluating Model Performance</h4>


```
def evaluateModel(yPredict,yTrue, classes):
    acc = np.round(accuracy_score(yTrue, yPredict), decimals=4)
    f1 = np.round(metrics.f1_score(yTrue, yPredict, average='macro'), decimals=4)
    print('Model Accuracy is: {}'.format(acc))
    print('Model F1 Score is: {}'.format(f1))
    cm = metrics.confusion_matrix(yTrue, yPredict)
    index = classes  
    columns = classes
    cm_df = pd.DataFrame(cm,columns,index)                      
    plt.figure(figsize=(10,6))  
    sns.heatmap(cm_df, annot=True)
```


```
evaluateModel(knn_optimal_predict,yTest_label,uniqueClasses)
```

    Model Accuracy is: 0.5925
    Model F1 Score is: 0.587



![png](output_46_1.png)


<h3>Logistic Regression Classifier</h3>
<h4>Setup Classifier</h4>


```
#Define ranges for pramaters
alpha_range = np.linspace(0.1,1,11)
l1_ratio = np.linspace(0.1,1,11)
log_param_grid = { 
    'alpha': alpha_range,
    'l1_ratio': l1_ratio
}
```


```
log_class = SGDClassifier(penalty='elasticnet', 
                          class_weight = 'balanced', 
                          max_iter = 1000, 
                          tol = 1e-3)
```

<h4>Grid-Search Cross-Validation</h4>


```
#Setup cross-validation model
log_cv = GridSearchCV(estimator = log_class,
                      param_grid =log_param_grid,
                      cv=cv_shuffle,
                      scoring = 'accuracy',
                      return_train_score = True)
```


```
log_cv_fit = log_cv.fit(Xtrain_pca, 
                        yTrain_label)
```


```
#print best parameters 
print ("The optimal Logistic Regression Model has an Alpha of {} and an L1-Ratio of {}".format(log_cv_fit.best_params_['alpha'], log_cv_fit.best_params_['l1_ratio']))
```

    The optimal Logistic Regression Model has an Alpha of 0.1 and an L1-Ratio of 0.1


<h4>Surface Plot</h4>


```
#Create dataframe of scores and parameter values
log_scores = pd.DataFrame(log_cv_fit.cv_results_)
```


```
#Find Optimal Prameters
log_opti = log_scores['rank_test_score'] == 1
log_optimal_run = log_scores[log_opti]
log_z = log_optimal_run['mean_test_score']
log_x = log_optimal_run['param_alpha']
log_y = log_optimal_run['param_l1_ratio']
```


```
log_optimal_run
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>mean_score_time</th>
      <th>mean_test_score</th>
      <th>mean_train_score</th>
      <th>param_alpha</th>
      <th>param_l1_ratio</th>
      <th>params</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split0_train_score</th>
      <th>...</th>
      <th>split7_test_score</th>
      <th>split7_train_score</th>
      <th>split8_test_score</th>
      <th>split8_train_score</th>
      <th>split9_test_score</th>
      <th>split9_train_score</th>
      <th>std_fit_time</th>
      <th>std_score_time</th>
      <th>std_test_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.114095</td>
      <td>0.000938</td>
      <td>0.881937</td>
      <td>0.882587</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>{u'alpha': 0.1, u'l1_ratio': 0.1}</td>
      <td>1</td>
      <td>0.89538</td>
      <td>0.884371</td>
      <td>...</td>
      <td>0.888435</td>
      <td>0.880459</td>
      <td>0.90068</td>
      <td>0.883482</td>
      <td>0.877551</td>
      <td>0.881668</td>
      <td>0.014332</td>
      <td>0.000769</td>
      <td>0.012483</td>
      <td>0.003024</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 32 columns</p>
</div>




```
#Set X,Y,Z Coordinates
x_log = log_scores['param_alpha'].values
y_log = log_scores['param_l1_ratio'].values
z_log = log_scores['mean_test_score'].values
```


```
#Plot 3D Surface plot
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(list(x_log), list(y_log), list(z_log), cmap=cm.coolwarm, alpha=0.7)
#Mark Optimal Paramters on Graph
ax.scatter3D(log_x, log_y, log_z, s= 200, marker = '^', alpha = 1)
ax.set_xlabel('Alpha')
ax.set_ylabel('L1-Ratio')
ax.set_zlabel('Validation Scores')
ax.dist=12
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5, forward= True)
ax.view_init(elev=20, azim=30)
plt.show()
```


![png](output_59_0.png)


<h4>Fit Optimal Model</h4>


```
#Model using optimal hyperparameter values
log_class_optimal = SGDClassifier(penalty='elasticnet',
                                  class_weight = 'balanced',
                                  alpha = 0.1,
                                  l1_ratio = 0.1,
                                 tol = 1e-3,
                                 max_iter = 1000)
```


```
#Fit model to training data
log_class_optimal.fit(Xtrain_pca, yTrain_label)
```




    SGDClassifier(alpha=0.1, average=False, class_weight='balanced',
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.1, learning_rate='optimal', loss='hinge', max_iter=1000,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
           power_t=0.5, random_state=None, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)




```
#Predict Using final model
log_optimal_predict = log_class_optimal.predict(Xtest_pca)
```

<h4>Evaluate Model Performace</h4>


```
evaluateModel(log_optimal_predict,yTest_label,uniqueClasses)
```

    Model Accuracy is: 0.622
    Model F1 Score is: 0.6103



![png](output_65_1.png)


<h4>Comments</h4>

The training model showed an average F1-Score of over 90%. When this model was used to predict calsses using the previously unseen testing dataset it only achieved an F1-Score of ~63%. This is a clear sign of overfitting and suggests that the model overl complex and does not generalise well. 

<h3>Support Vector Machine Classifier</h3>

<h4>Setup Classifier</h4>


```
#Define ranges for pramaters
gamma_range = np.linspace(2e-5,2e-20,20)
c_range = np.linspace(0.001,100,20)
svm_param_grid = { 
    'gamma': gamma_range,
    'C': c_range
}
```


```
#Setup classifier
svm_class = svm.SVC(class_weight = 'balanced',
                    kernel = 'rbf',
                    decision_function_shape = 'ova',
                    random_state = 42)
```

<h4>Grid Search CV</h4>


```
#Setup cross-validation model
svm_cv = GridSearchCV(estimator = svm_class,
                      param_grid =svm_param_grid,
                      cv=cv_shuffle,
                      scoring = "accuracy",
                      return_train_score = True)
```


```
#Hyperparameter tuning using GridSearch CV
svm_cv_fit = svm_cv.fit(Xtrain_pca, 
                        yTrain_label)
```


```
#print best parameters 
print ("The optimal SVM Model has a Gamma of {} and a C of {}".format(svm_cv_fit.best_params_['gamma'], svm_cv_fit.best_params_['C']))
```

    The optimal SVM Model has a Gamma of 0.112 and a C of 1.0


<h4>Surface Plot</h4>


```
#Create dataframe of scores and parameter values
svm_scores = pd.DataFrame(svm_cv_fit.cv_results_)
```


```
#Find Optimal Prameters
svm_opti = svm_scores['rank_test_score'] == 1
svm_optimal_run = svm_scores[svm_opti]
svm_z = svm_optimal_run['mean_test_score']
svm_x = svm_optimal_run['param_gamma']
svm_y = svm_optimal_run['param_C']
```


```
#Plot 3D Surface plot
fig = plt.figure()
ax = Axes3D(fig)
x_svm = svm_scores['param_gamma'].values
y_svm = svm_scores['param_C'].values
z_svm = svm_scores['mean_test_score'].values
ax.plot_trisurf(list(x_svm), list(y_svm), list(z_svm), cmap=cm.coolwarm, alpha=0.7)
#Mark Optimal Paramters on Graph
ax.scatter3D(svm_x, svm_y, svm_z, s= 100, alpha = 0.7)
#Graph Labels
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_zlabel('Validation Scores')
ax.dist=12
fig = plt.gcf()
fig.set_size_inches(20, 10.5, forward= True)
ax.view_init(elev=60, azim=20)
plt.show()
```


![png](output_78_0.png)


<h4>Describe Surface-Plot</h4>

The Surface plot above shows how varying paramers **gamma** and **C** affect validation scores. 
We can see from the graph that in this instance, both gamma and C seem to have a positive proportional relationship with the validation score. That is as both increase, validation F1-Score also increases.

<h4>Fit Optimal Model</h4>


```
#Setup classifier
svm_class_optimal = svm.SVC(class_weight = 'balanced',
                            kernel = 'rbf',
                            decision_function_shape = 'ovo',
                            random_state = 42,
                            gamma = 0.112,
                            C = 1,
                            max_iter = 100)
```


```
#Fit model to training data
svm_class_optimal.fit(Xtrain_pca, yTrain_label)
```

    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/svm/base.py:244: ConvergenceWarning: Solver terminated early (max_iter=100).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
      % self.max_iter, ConvergenceWarning)





    SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovo', degree=3, gamma=0.112, kernel='rbf',
      max_iter=100, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)




```
#Predict Using final model
svm_optimal_predict = svm_class_optimal.predict(Xtest_pca)
```

<h4>Evaluate Model Performance</h4>


```
#Confusion Matrix
evaluateModel(svm_optimal_predict,yTest_label,uniqueClasses)
```

    Model Accuracy is: 0.5901
    Model F1 Score is: 0.542



![png](output_85_1.png)


<h4>Comments</h4>

Once again huge difference between training and testing score. Sign of overfitting!

<h3>Random Forest Classifier</h3>

<h4>Setup Classifier</h4>


```
#Define ranges for pramaters
depth_range = np.linspace(100, 1000, 10)
tree_range = np.linspace(100, 1000, 10)
rf_param_grid = { 
    'n_estimators': tree_range,
    'max_depth': depth_range
}
```


```
rf_class = RandomForestClassifier(bootstrap=True,
                                  class_weight = 'balanced_subsample',
                                  random_state = 42,
                                 oob_score = True, 
                                 n_jobs = -1)
```

<h4>Grid Search CV</h4>


```
#Setup cross-validation model
rf_cv = GridSearchCV(estimator = rf_class,
                      param_grid =rf_param_grid,
                      cv=cv_shuffle,
                      scoring = f1,
                      return_train_score = True)
```


```
#Hyperparameter tuning using GridSearch CV
rf_cv_fit = rf_cv.fit(Xtrain_pca, 
                        yTrain_label)
```

    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/model_selection/_validation.py:542: FutureWarning: From version 0.22, errors during fit will result in a cross validation score of NaN by default. Use error_score='raise' if you want an exception raised or error_score=np.nan to adopt the behavior from version 0.22.
      FutureWarning)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-203-bfc6ae973f9d> in <module>()
          1 #Hyperparameter tuning using GridSearch CV
          2 rf_cv_fit = rf_cv.fit(Xtrain_pca, 
    ----> 3                         yTrain_label)
    

    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/model_selection/_search.pyc in fit(self, X, y, groups, **fit_params)
        720                 return results_container[0]
        721 
    --> 722             self._run_search(evaluate_candidates)
        723 
        724         results = results_container[0]


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/model_selection/_search.pyc in _run_search(self, evaluate_candidates)
       1189     def _run_search(self, evaluate_candidates):
       1190         """Search all candidates in param_grid"""
    -> 1191         evaluate_candidates(ParameterGrid(self.param_grid))
       1192 
       1193 


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/model_selection/_search.pyc in evaluate_candidates(candidate_params)
        709                                for parameters, (train, test)
        710                                in product(candidate_params,
    --> 711                                           cv.split(X, y, groups)))
        712 
        713                 all_candidate_params.extend(candidate_params)


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self, iterable)
        915             # remaining jobs.
        916             self._iterating = False
    --> 917             if self.dispatch_one_batch(iterator):
        918                 self._iterating = self._original_iterator is not None
        919 


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in dispatch_one_batch(self, iterator)
        757                 return False
        758             else:
    --> 759                 self._dispatch(tasks)
        760                 return True
        761 


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in _dispatch(self, batch)
        714         with self._lock:
        715             job_idx = len(self._jobs)
    --> 716             job = self._backend.apply_async(batch, callback=cb)
        717             # A job can complete so quickly than its callback is
        718             # called before we get here, causing self._jobs to


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/_parallel_backends.pyc in apply_async(self, func, callback)
        180     def apply_async(self, func, callback=None):
        181         """Schedule a func to be run"""
    --> 182         result = ImmediateResult(func)
        183         if callback:
        184             callback(result)


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/_parallel_backends.pyc in __init__(self, batch)
        547         # Don't delay the application, to avoid keeping the input
        548         # arguments in memory
    --> 549         self.results = batch()
        550 
        551     def get(self):


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/externals/joblib/parallel.pyc in __call__(self)
        223         with parallel_backend(self._backend, n_jobs=self._n_jobs):
        224             return [func(*args, **kwargs)
    --> 225                     for func, args, kwargs in self.items]
        226 
        227     def __len__(self):


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/model_selection/_validation.pyc in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, error_score)
        526             estimator.fit(X_train, **fit_params)
        527         else:
    --> 528             estimator.fit(X_train, y_train, **fit_params)
        529 
        530     except Exception as e:


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/ensemble/forest.pyc in fit(self, X, y, sample_weight)
        286 
        287         # Check parameters
    --> 288         self._validate_estimator()
        289 
        290         if not self.bootstrap and self.oob_score:


    /Users/ragyibrahim/anaconda2/lib/python2.7/site-packages/sklearn/ensemble/base.pyc in _validate_estimator(self, default)
        104         if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
        105             raise ValueError("n_estimators must be an integer, "
    --> 106                              "got {0}.".format(type(self.n_estimators)))
        107 
        108         if self.n_estimators <= 0:


    ValueError: n_estimators must be an integer, got <type 'numpy.float64'>.



```
#print best parameters 
print ("The optimal Random Forest Model has {} Trees with a Tree Depth of {}".format(rf_cv_fit.best_params_['max_depth'], rf_cv_fit.best_params_['n_estimators']))
```

    The optimal Random Forest Model has 300 Trees with a Tree Depth of 700


<h4>Surface Plot</h4>


```
#Create dataframe of scores and parameter values
rf_scores = pd.DataFrame(rf_cv_fit.cv_results_)
```


```
#Find Optimal Prameters
rf_opti = rf_scores['rank_test_score'] == 1
rf_optimal_run = rf_scores[rf_opti]
rf_z = rf_optimal_run['mean_test_score']
rf_x = rf_optimal_run['param_max_depth']
rf_y = rf_optimal_run['param_n_estimators']
```


```
rf_optimal_run
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>mean_score_time</th>
      <th>mean_test_score</th>
      <th>mean_train_score</th>
      <th>param_max_depth</th>
      <th>param_n_estimators</th>
      <th>params</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split0_train_score</th>
      <th>...</th>
      <th>split7_test_score</th>
      <th>split7_train_score</th>
      <th>split8_test_score</th>
      <th>split8_train_score</th>
      <th>split9_test_score</th>
      <th>split9_train_score</th>
      <th>std_fit_time</th>
      <th>std_score_time</th>
      <th>std_test_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>6.047658</td>
      <td>0.287613</td>
      <td>0.95284</td>
      <td>1.0</td>
      <td>300</td>
      <td>700</td>
      <td>{u'n_estimators': 700, u'max_depth': 300}</td>
      <td>1</td>
      <td>0.965506</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.948159</td>
      <td>1.0</td>
      <td>0.953079</td>
      <td>1.0</td>
      <td>0.950236</td>
      <td>1.0</td>
      <td>0.354524</td>
      <td>0.058531</td>
      <td>0.007337</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.371097</td>
      <td>0.346319</td>
      <td>0.95284</td>
      <td>1.0</td>
      <td>500</td>
      <td>700</td>
      <td>{u'n_estimators': 700, u'max_depth': 500}</td>
      <td>1</td>
      <td>0.965506</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.948159</td>
      <td>1.0</td>
      <td>0.953079</td>
      <td>1.0</td>
      <td>0.950236</td>
      <td>1.0</td>
      <td>0.361720</td>
      <td>0.012973</td>
      <td>0.007337</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.033265</td>
      <td>0.323326</td>
      <td>0.95284</td>
      <td>1.0</td>
      <td>600</td>
      <td>700</td>
      <td>{u'n_estimators': 700, u'max_depth': 600}</td>
      <td>1</td>
      <td>0.965506</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.948159</td>
      <td>1.0</td>
      <td>0.953079</td>
      <td>1.0</td>
      <td>0.950236</td>
      <td>1.0</td>
      <td>0.251601</td>
      <td>0.052656</td>
      <td>0.007337</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>




```
#Plot 3D Surface plot
fig = plt.figure()
ax = Axes3D(fig)
x_rf = rf_scores['param_max_depth'].values
y_rf = rf_scores['param_n_estimators'].values
z_rf = rf_scores['mean_test_score'].values
ax.plot_trisurf(list(x_rf), list(y_rf), list(z_rf), cmap=cm.coolwarm, alpha=0.7)
#Mark Optimal Paramters on Graph
ax.scatter3D(rf_x, rf_y, rf_z, s= 1000, alpha = 0.3)
#Graph Labels
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_zlabel('Validation Scores')
ax.dist=12
fig = plt.gcf()
fig.set_size_inches(20, 10.5, forward= True)
ax.view_init(elev=60, azim=130)
plt.show()
```


![png](output_99_0.png)


<h4>Describe Surfcae-Plot</h4>

Here we have a very interesting scenario - there are 3 hyperprameter combinations that are indistiguishable from one another in their performance. Whats more interesting is that they show that, in this instance, the tree depth is some-what of a redundant paramter. By setting the number of trees to **700** any combination of tree depth will generate a *perfect* training score, which is a strong indication of overfitting. 

The relationship described by this plot is a linear proportionality between both parameters and the validation score.

<h4>Fit Optimal Model</h4>

Given that any <code>max_depth</code> corresponds to an optimal solution i will choose the least complex (i.e. <code>max_depth = 300</code>) in the hope that the model will generalise better the test dataset


```
rf_class_optimal = RandomForestClassifier(bootstrap=True,
                                  class_weight = 'balanced_subsample',
                                  random_state = 42,
                                  oob_score = True, 
                                  n_jobs = -1,
                                  max_depth = 300,
                                  n_estimators = 700)
```


```
#Fit model to training data
rf_class_optimal.fit(Xtrain_pca, yTrain_label)
```




    RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
                criterion='gini', max_depth=300, max_features='auto',
                max_leaf_nodes=None, min_impurity_decrease=0.0,
                min_impurity_split=None, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=700, n_jobs=-1, oob_score=True, random_state=42,
                verbose=0, warm_start=False)




```
#Predict Using final model
rf_optimal_predict = rf_class_optimal.predict(Xtest_pca)
```


```
#Confusion Matrix
evaluateModel(rf_optimal_predict,yTest_label,uniqueClasses)
```

    Model Accuracy is: 0.5568
    Model F1 Score is: 0.5577



![png](output_105_1.png)


This notebook is based on the word done in: 

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.


```

```
