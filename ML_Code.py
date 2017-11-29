
# coding: utf-8

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
get_ipython().magic('matplotlib inline')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression# to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.neural_network import MLPClassifier #for MLP classifier
from sklearn.metrics import confusion_matrix
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import tensorflow as tf


# In[2]:


data = pd.read_csv("/Users/niteshjain/Desktop/data.csv",header=0)# here header 0 means the 0 th row is our coloumn 
                                                # header in data


# In[3]:


# have a look at the data
print(data.head(2))


# In[4]:


data.drop("Unnamed: 32",axis=1,inplace=True)
data.drop("id",axis=1,inplace=True)
data.columns


# In[5]:


features_all = list(data.columns[1:31])
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:21])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)


# In[6]:


df=data
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})


# In[7]:


for i in range (0 , len(features_mean)):
    plt.figure(figsize=(1,5))
    sns.boxplot(x=df.diagnosis,y=features_mean[i],data=df)
#plt.boxplot('diagnosis',y='texture_mean',data=data,ax=ax1)


# In[8]:


for i in range (0 , len(features_se)):
    plt.figure(figsize=(1,5))
    sns.boxplot(x=df.diagnosis,y=features_se[i],data=df)


# In[9]:


for i in range (0 , len(features_worst)):
    plt.figure(figsize=(1,5))
    sns.boxplot(x=df.diagnosis,y=features_worst[i],data=df)


# In[10]:


sl_features = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean',
               'radius_se','perimeter_se','area_se','compactness_se','concave points_se',
              'radius_worst','perimeter_worst','area_worst','compactness_worst','concavity_worst','concave points_worst']


# In[11]:


corr_me = data[sl_features].corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_me, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           xticklabels= sl_features, yticklabels= sl_features,
           cmap= 'coolwarm')


# In[12]:


prediction_var_me = ['radius_mean','concavity_mean','concave points_mean','area_se','concave points_worst']


# In[13]:


#now split our data into train and test
train_me, test_me = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train_me.shape)
print(test_me.shape)


# In[14]:


train_me_X = train_me[prediction_var_me]# taking the training data input 
train_me_y=train_me.diagnosis# This is output of our training data
# same we have to do for test
test_me_X= test_me[prediction_var_me] # taking test data inputs
test_me_y =test_me.diagnosis   #output value of test dat

print(test_me_X.shape)


# In[15]:


RFC_model_me=RandomForestClassifier(n_estimators=100)# a simple random forest model
SVM_model_me=svm.SVC()
MLP_model_me=MLPClassifier(solver='lbfgs',alpha=5,hidden_layer_sizes=(500,),random_state=10)
KNN_model_me=KNeighborsClassifier()
LR_model_me = LinearRegression()
LGR_model_me = LogisticRegression()


# In[16]:


RFC_model_me.fit(train_me_X,train_me_y)# now fit our model for traiing data
SVM_model_me.fit(train_me_X,train_me_y)
MLP_model_me.fit(train_me_X,train_me_y)
KNN_model_me.fit(train_me_X,train_me_y)
LR_model_me.fit(train_me_X,train_me_y)
LGR_model_me.fit(train_me_X,train_me_y)


# In[17]:


RFC_prediction_me=RFC_model_me.predict(test_me_X)
SVM_prediction_me=SVM_model_me.predict(test_me_X)
MLP_prediction_me=MLP_model_me.predict(test_me_X)
KNN_prediction_me=KNN_model_me.predict(test_me_X)
LR_prediction_me = LR_model_me.predict(test_me_X)
LGR_prediction_me = LGR_model_me.predict(test_me_X)
# predict for the test data
# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs


# In[18]:


RFaccuracy_me=metrics.accuracy_score(RFC_prediction_me,test_me_y)
SVMaccuracy_me=metrics.accuracy_score(SVM_prediction_me,test_me_y)
MLPaccuracy_me=metrics.accuracy_score(MLP_prediction_me,test_me_y)
KNNaccuracy_me=metrics.accuracy_score(KNN_prediction_me,test_me_y)
#LRaccuracy_me = metrics.accuracy_score(LR_prediction_me,test_me_y)
LGRaccuracy_me=metrics.accuracy_score(LGR_prediction_me,test_me_y)
print("Random Forest Accuracy with mean features is" , RFaccuracy_me)
print("SVM accuracy with mean features is" , SVMaccuracy_me)
print('MLP Classfier accuracy with me features is', MLPaccuracy_me)
print('KNN classifier accuracy with me feaures is', KNNaccuracy_me)
#print('Linear regression accuracy with me feaures is', LRaccuracy_me)
print('Logistic regression accuracy with me feaures is', LGRaccuracy_me)
# to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values


# In[19]:


from sklearn.feature_selection import RFE
modeltry = RandomForestClassifier(n_estimators=100)
# create the RFE model and select 3 attributes
rfe = RFE(modeltry, 3)
rfe = rfe.fit(train_me_X,train_me_y)
rfe_pred = rfe.predict(test_me_X)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
print(metrics.accuracy_score(rfe_pred,test_me_y))


# In[20]:



# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
 #                          n_redundant=2, n_repeated=0, n_classes=8,
  #                         n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
RF_rfecv = RFECV(estimator=modeltry, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
RF_rfecv.fit(train_me_X,train_me_y)
features=RF_rfecv.get_support(indices=False)
RF_rfecv_predict=RF_rfecv.predict(test_me_X)
RF_rfecv_accuracy = metrics.accuracy_score(RF_rfecv_predict,test_me_y)



print(features)
print("Optimal number of features : %d" % RF_rfecv.n_features_)
print(RF_rfecv_accuracy)


# In[21]:


color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix


# In[22]:


color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
colors = data["diagnosis"].map(lambda x: color_function.get(x))
plt.scatter(features_all[1],features_all[17],c=colors,data=df)


# In[23]:


train_DNN, test_DNN = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train_DNN.shape)
print(test_DNN.shape)
train_DNN_X = train_DNN[features_all]# taking the training data input 
train_DNN_y=train_DNN.diagnosis# This is output of our training data
# same we have to do for test
test_DNN_X= test_DNN[features_all] # taking test data inputs
test_DNN_y =test_DNN.diagnosis   #output value of test dat
print(test_DNN_X.shape)


# In[24]:


import tensorflow as tf
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_DNN_X)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[60,120,60], n_classes=2)
classifier.fit(train_DNN_X,train_DNN_y,steps=2000)

accuracy_score = classifier.evaluate(x=test_DNN_X,
                                     y=test_DNN_y)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

scores = cross_val_score(estimator=classifier,
X=data,
y=data.diagnosis,
scoring = 'accuracy',
cv=5,
fit_params={'steps': 2000},
)


# In[50]:


def classification_model(model,data,prediction_input,output):
    
    model.fit(data[prediction_input],data[output]) #Here we fit the model using training set
  
    
    predictions = model.predict(data[prediction_input])
    cmatrix = confusion_matrix(data[output], predictions)
    
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
 
    
    kf = KFold(data.shape[0], n_folds=5)
    
    error = []
    for train, test in kf:
        
        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data
        # here iloc[train,:] means all row in train in kf amd the all columns
        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train
        # Training the algorithm using the predictors and target.
        model.fit(train_X, train_y)
    
        # now do this for test data also
        test_X=data[prediction_input].iloc[test,:]
        test_y=data[output].iloc[test]
        error.append(model.score(test_X,test_y))
        predict=model.predict(test_X)
        
        # printing the score 
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

        
    return cmatrix    


# In[56]:


#train_RFC, test_RFC = train_test_split(data, test_size = 0.3)
RFC_model=RandomForestClassifier(n_estimators=100)
RFC_model_var=['radius_mean']
RFC_outcome_var='diagnosis'
RFC_cmatrix=classification_model(RFC_model,data,RFC_model_var,RFC_outcome_var)

plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(RFC_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')


# In[27]:


#train_RFC, test_RFC = train_test_split(data, test_size = 0.3)
RFC_model=RandomForestClassifier(n_estimators=100)
RFC_model_var=sl_features
RFC_outcome_var='diagnosis'
classification_model(RFC_model,data,RFC_model_var,RFC_outcome_var)


# In[58]:


SVM_model=svm.SVC()
SVM_var=['radius_mean']
SVM_output='diagnosis'
SVM_cmatrix=classification_model(SVM_model,data,SVM_var,SVM_output)

plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(SVM_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')


# In[59]:


KNN_model=KNeighborsClassifier()
KNN_var=features_all
KNN_output='diagnosis'
KNN_cmatrix=classification_model(KNN_model,data,KNN_var,KNN_output)

plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(KNN_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')


# In[60]:


LGR_model = LogisticRegression()
LGR_var=features_all
LGR_output='diagnosis'
LGR_cmatrix=classification_model(LGR_model,data,LGR_var,LGR_output)

plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(LGR_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')


# In[61]:


MLP_model=MLPClassifier(solver='lbfgs',alpha=5,hidden_layer_sizes=(500,),random_state=10)
MLP_var=features_all
MLP_output='diagnosis'
MLP_cmatrix=classification_model(MLP_model,data,MLP_var,MLP_output)
plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(MLP_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')


# In[62]:


GNB_model=GaussianNB()
GNB_var=features_all
GNB_output='diagnosis'
GNB_cmatrix=classification_model(GNB_model,data,GNB_var,GNB_output)

plt.rcParams['figure.figsize']=(14,8)
ax = plt.axes()
sns.heatmap(GNB_cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})
ax.set_title('Confusion Matrix')

