#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as mtp
import numpy as nm


# # 0. Data info
#  
# ### The attributes related with eating habits are:
# - Frequent consumption of high caloric food (FAVC) 
# - Frequency of consumption of vegetables (FCVC)
# - Number of main meals (NCP)
# - Consumption of food between meals (CAEC)
# - Consumption of water daily (CH20)
# - Consumption of alcohol (CALC) 
# 
# ### The attributes related with the physical condition are: 
# - Calories consumption monitoring (SCC)
# - Physical activity frequency (FAF)
# - Time using technology devices (TUE)
# - Transportation used (MTRANS)
# - other variables obtained were: Gender, Age, Height and Weight
# 
# # 
# 
# NObesity was created with the values of: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III
# 
# - Underweight Less than 18.5
# - Normal 18.5 to 24.9
# - Overweight 25.0 to 29.9
# - Obesity I 30.0 to 34.9
# - Obesity II 35.0 to 39.9
# - Obesity III Higher than 40

# # 1. Loading Data

# In[6]:


data_set= pd.read_csv('Desktop/ObesityDataSet_raw_and_data_sinthetic.csv')


# In[7]:


data_set


# # 2. Data Wrangling
# 
# ##      2.1. Data statistics

# In[8]:


data_set.shape


# In[9]:


data_set.info()


# In[10]:


data_set.describe()


# ## 2.2. Dropping duplicate rows

# In[11]:


data_set = data_set.drop_duplicates()


# In[12]:


data_set.shape


# ## 2.3. Quantifying missing data

# In[13]:


data_set.isnull().sum()


# In[14]:


data_set.isnull().mean()


# ## 2.4.  Handling imbalanced data

# In[15]:


print(data_set['NObeyesdad'].value_counts(), end='\n\n\n')
columns = ['Obesity_Type_I', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_II', 'Normal_Weight', 'Overweight_Level_I', 'Insufficient_Weight']
mtp.bar(columns,data_set['NObeyesdad'].value_counts(),width=0.8)
mtp.xticks(rotation=90)
mtp.show()


# In[ ]:





# # 3. Descriptive statistics

# In[16]:


average = data_set.iloc[:, [1,2,3,6,7,10,12,13]].mean()
print("\033[1m Mean: \033[0m \n\n",average, end = '\n\n\n')

med = data_set.iloc[:, [1,2,3,6,7,10,12,13]].median()
print("\033[1m Median: \033[0m \n\n", med, end = '\n\n\n')

mode = data_set.mode()
print("\033[1m Mode: \033[0m \n\n", mode, end = '\n\n\n')

standard_dev = data_set.iloc[:, [1,2,3,6,7,10,12,13]].std()
print("\033[1m Standard deviation: \033[0m \n\n", standard_dev, end = '\n\n\n')

var = data_set.iloc[:, [1,2,3,6,7,10,12,13]].var()
print("\033[1m Variance: \033[0m \n\n", var, end = '\n\n\n')

ske = data_set.iloc[:, [1,2,3,6,7,10,12,13]].var()
print("\033[1m Skewness: \033[0m \n\n", ske, end = '\n\n\n')

#import seaborn as sns
#import matplotlib.pyplot as plt

#%matplotlib inline

#sns.set(style="whitegrid")
#plt.figure(figsize=(10,8))
#vals = [1, 2, 3, 6, 7, 10, 12, 13]
#for i in vals:
#    sns.boxplot(data=data_set.iloc[:,i], orient="v")


# ## 3.1.  Correlations

# In[17]:


from random import randint
cols = []
for i in range(15):
    col = '#%06X' % randint(0, 0xFFFFFF)
    cols.append(col)
    data_set.plot(x=data_set.columns[i],y="NObeyesdad", kind="scatter", color=col)


# In[18]:


data_set.corr(method='pearson')


# In[19]:


from pandas.plotting import scatter_matrix
from matplotlib import cm
import matplotlib.pyplot as plt
feature_names = data_set.columns
X = data_set[feature_names]
y = data_set['NObeyesdad']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c = "purple", marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(15,15), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('NObeyesdad_scatter_matrix')


# ![image.png](attachment:image.png)

# ## 3.2.  Separating data

# In[20]:


x= data_set.iloc[:,:-1].values


# In[21]:


x


# In[23]:


y= data_set.iloc[:,16].values  
y
xx = nm.array(data_set.iloc[:,16].values)
uniqueY = nm.unique(xx)


# ## 3.3. Encoding categorical variables

# In[24]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])
x[:, 4]= label_encoder_x.fit_transform(x[:, 4]) 
x[:, 5]= label_encoder_x.fit_transform(x[:, 5]) 
x[:, 8]= label_encoder_x.fit_transform(x[:, 8]) 
x[:, 9]= label_encoder_x.fit_transform(x[:, 9]) 
x[:, 11]= label_encoder_x.fit_transform(x[:, 11])  
x[:, 14]= label_encoder_x.fit_transform(x[:, 14])  
x[:, 15]= label_encoder_x.fit_transform(x[:, 15])  
x[0]

x_temp = x


# In[25]:


labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y) 
y


# In[26]:


onehot_encoder= OneHotEncoder()    

d= onehot_encoder.fit_transform(x[:, [0, 4, 5, 8, 9, 11, 14, 15]]).toarray()

d

tempTable = x[:, [1,2,3,6,7,10,12,13]]

prepTable = nm.append(tempTable, d, axis=1)

prepTable


# In[27]:


x[:, [0, 4, 5, 8, 9, 11, 14, 15]]


# ## 3.4.  Separating data into train and test sets

# In[28]:


from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test= train_test_split(prepTable, y, test_size= 0.2, random_state=46)


# ## 3.5. Standardizing the features

# In[29]:


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
st_x= StandardScaler()  
data = pd.DataFrame(x_train)
forReduce = pd.DataFrame(prepTable)
data


# In[30]:


from sklearn.preprocessing import StandardScaler

st_x= StandardScaler()  

x_train_all= st_x.fit_transform(x_train)

x_test_all= st_x.transform(x_test)


# In[31]:


ct = ColumnTransformer([('somename', st_x, [0, 2])], remainder='passthrough')

tmp = ct.fit_transform(data)

print(tmp, end='\n\n')

print(tmp[0])

x_train_selective = tmp

data = pd.DataFrame(x_test)

ct = ColumnTransformer([('somename', st_x, [0, 2])], remainder='passthrough')

tmp = ct.fit_transform(data)

x_test_selective = tmp
#x_train[:,0]= st_x.fit_transform([x_train[:,0]])
#x_train[:,2]= st_x.fit_transform(x_train[:,2])
#x_tmp = st_x.fit_transform(x_train)
#x_tmp
#x_test= st_x.transform(x_test)


# In[32]:


x_train_selective = nm.array(x_train_selective, dtype=float)
x_test_selective = nm.array(x_test_selective, dtype=float)


# # 4. Model Evaluation

# ## 4.1. Cross-Validating Models

# In[34]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

features = x_temp
target = y

# Create standardizer
standardizer = StandardScaler()

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(x_temp, y, test_size= 0.1, random_state=1)

# Fit standardizer to training set
standardizer.fit(features_train)

# Apply to both training and test sets
#features_train_std = standardizer.transform(features_train)
#features_test_std = standardizer.transform(features_test)

logit = DecisionTreeClassifier(random_state=0)

# Create a pipeline
pipeline = make_pipeline(standardizer, logit)

# Create k-Fold cross-validation
kf = KFold(n_splits=20, shuffle=True, random_state=1)

# Do k-fold cross-validation
cv_results = cross_val_score(pipeline, features_train, target_train, cv=kf, scoring="accuracy", n_jobs=-1)


# In[35]:


cv_results


# In[36]:


# Calculate mean
cv_results.mean()


# ## 4.2. Creating a Baseline Classification Model

# In[38]:


from sklearn.dummy import DummyClassifier

# Create dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)
# "Train" model
dummy.fit(features_train, target_train)
# Get accuracy score
dummy.score(features_test, target_test)


# In[39]:


# Load library
from sklearn.ensemble import RandomForestClassifier
# Create classifier
classifier = RandomForestClassifier()
# Train model
classifier.fit(features_train, target_train)
# Get accuracy score
classifier.score(features_test, target_test)


# ## 4.3. Evaluating Multiclass Classifier Predictions

# In[40]:


from sklearn.linear_model import LogisticRegression

# Create logistic regression
logit = DecisionTreeClassifier(random_state=0)
# Cross-validate model using accuracy
cross_val_score(logit, features, target, scoring='accuracy')


# ## 4.4. Visualizing a Classifier’s Performance

# In[42]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
class_names = data_set.iloc[:,-1].unique()

# Create logistic regression
classifier = DecisionTreeClassifier() #LogisticRegression()

# Train model and make predictions
target_predicted = classifier.fit(features_train, target_train).predict(features_test)

# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)


dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[43]:


data_set.iloc[:,-1].unique()


# ## 4.5. Visualizing the Effect of Training Set Size

# In[44]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), features, target, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=nm.linspace(0.01,1.0,50))

# Create means and standard deviations of training set scores
train_mean = nm.mean(train_scores, axis=1)
train_std = nm.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = nm.mean(test_scores, axis=1)
test_std = nm.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#685214", label="Training score")
plt.plot(train_sizes, test_mean, color="#508562", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ## 4.6.Visualizing the Effect of Hyperparameter Values

# In[45]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

# Create range of values for parameter
param_range = nm.arange(1, 250, 2)

train_scores, test_scores = validation_curve(RandomForestClassifier(), features, target, param_name="n_estimators",param_range=param_range, cv=3, scoring='accuracy', n_jobs=-1)

# Create means and standard deviations of training set scores
train_mean = nm.mean(train_scores, axis=1)
train_std = nm.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = nm.mean(test_scores, axis=1)
test_std = nm.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()


# # 5. Classification

# ## 5.1. KNeighbors Classifier

# In[46]:


from sklearn.neighbors import KNeighborsClassifier
neigh_all = KNeighborsClassifier(n_neighbors=3)
neigh_selective = KNeighborsClassifier(n_neighbors=3)
neigh_all.fit(x_train_all, y_train)
neigh_selective.fit(x_train_selective, y_train)


# In[47]:


x_test_all[0]


# In[48]:


y_test[0]


# In[49]:


print(neigh_all.predict([x_test_all[0],x_test_all[1],x_test_all[2]]))


# In[50]:


num = 0
for i in range(len(x_test_all)):
    if(y_test[i]==neigh_all.predict([x_test_all[i]])[0]):
        num = num + 1
    # print(y_test[i], neigh.predict([x_test[i]]))
print(num, len(x_test_all), num/len(x_test_all))


# In[51]:


num = 0
for i in range(len(x_test_selective)):
    if(y_test[i]==neigh_selective.predict([x_test_selective[i]])[0]):
        num = num + 1
print(num, len(x_test_selective), num/len(x_test_selective))


# ## 5.2. Radius Neighbors Classifier

# In[52]:


from sklearn.neighbors import RadiusNeighborsClassifier
neigh2_all = RadiusNeighborsClassifier(radius=30.0)
neigh2_selective = RadiusNeighborsClassifier(radius=30.0)
neigh2_all.fit(x_train_all, y_train)
neigh2_selective.fit(x_train_selective, y_train)
num = 0
for i in range(len(x_test_all)):
    if(y_test[i]==neigh2_all.predict([x_test_all[i]])[0]):
        num = num + 1
    # print(y_test[i], neigh.predict([x_test[i]]))
print(num, len(x_test_all), num/len(x_test_all))


# In[53]:


num = 0
for i in range(len(x_test_selective)):
    if(y_test[i]==neigh2_selective.predict([x_test_selective[i]])[0]):
        num = num + 1
    # print(y_test[i], neigh.predict([x_test[i]]))
print(num, len(x_test_selective), num/len(x_test_selective))


# ## 5.3. Extra Tree Classifier

# In[54]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
extra_tree = ExtraTreeClassifier(random_state=0)
cls_all = BaggingClassifier(extra_tree, random_state=0).fit(x_train_all, y_train)
cls_all.score(x_test_all, y_test)


# In[55]:


cls_selective = BaggingClassifier(extra_tree, random_state=0).fit(x_train_selective, y_train)
cls_selective.score(x_test_selective, y_test)


# ## 5.4. Decision Tree Classifier

# In[56]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=0)
cls2_all = BaggingClassifier(dec_tree, random_state=0).fit(x_train_all, y_train)
cls2_all.score(x_test_all, y_test)


# In[57]:


cls2_selective = BaggingClassifier(dec_tree, random_state=0).fit(x_train_selective, y_train)
cls2_selective.score(x_test_selective, y_test)


# ## 5.5. Random Forest Classifier

# In[58]:


from sklearn.ensemble import RandomForestClassifier
rand_forest = DecisionTreeClassifier(random_state=0)
cls3_all = BaggingClassifier(rand_forest, random_state=0).fit(x_train_all, y_train)
cls3_all.score(x_test_all, y_test)


# In[59]:


cls3_selective = BaggingClassifier(rand_forest, random_state=0).fit(x_train_selective, y_train)
cls3_selective.score(x_test_selective, y_test)


# ## 5.6. MLP Classifier

# In[60]:


from sklearn.neural_network import MLPClassifier
clf4_all = MLPClassifier(random_state=1, max_iter=3000).fit(x_train_all, y_train)
clf4_all.score(x_test_all, y_test)


# In[61]:


clf4_selective = MLPClassifier(random_state=1, max_iter=3000).fit(x_train_selective, y_train)
clf4_selective.score(x_test_selective, y_test)


# # 6. Reducing data

# ## 6.1. Univariate Selection

# In[62]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = data_set
X = x  #independent columns
y = y  #target column
#apply SelectKBest class to extract top 10 best features
data_set
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data_set.iloc[:,0:16].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(16,'Score'))  #print 10 best features


# ## 6.2. Feature Importance

# In[63]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=data_set.iloc[:,0:16].columns)
feat_importances.nlargest(16).plot(kind='barh')
plt.show()


# ## 6.3. Correlation Matrix with Heatmap

# In[64]:


import seaborn as sns
#get correlations of each features in dataset
#corrTable = nm.c_[x[:,0:16], y[:]]

#dataframe = pd.DataFrame(corrTable, columns = data_set.columns)

corrmat = data_set.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data_set[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ## 6.4.  Removing Irrelevant Features for Classification

# In[65]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# Select two features with highest chi-squared statistics
chi2_selector = SelectKBest(chi2, k=5)
features_kbest = chi2_selector.fit_transform(x_temp, y)
# Show results
print("Original number of features:", x_temp.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
features_kbest #gender, age, weight, family, ...


# ## 6.5. Recursively Eliminating Features

# In[66]:


from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import linear_model

# Create a linear regression
ols = linear_model.LinearRegression()
# Recursively eliminate features
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(x_temp, y)
rfecv.transform(x_temp)


# In[67]:


rfecv.n_features_


# In[68]:


rfecv.support_


# In[69]:


# Rank features best (1) to worst
rfecv.ranking_


# In[70]:


featureScores = pd.concat([dfcolumns,pd.DataFrame(rfecv.ranking_)],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(16,'Score'))


# ## 6.6. Results - reduced dataset

# In[71]:


#REMOVED: SMOKE, FAVC, TUE, CH2O, CAEC, FAVC
# 5, 7, 13, 14, 15, 16, 17, 18, 19, 20
newData = forReduce.drop([5, 7, 13, 14, 15, 16, 17, 18, 19, 20], axis=1)
newData


# In[72]:


x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(newData, y, test_size= 0.2, random_state=46)


# In[73]:


x_train_red_st = st_x.fit_transform(x_train_reduced)
x_test_red_st = st_x.fit_transform(x_test_reduced)
x_train_red_st


# In[74]:


dec_tree = DecisionTreeClassifier(random_state=0)
cls2_all = BaggingClassifier(dec_tree, random_state=0).fit(x_train_red_st, y_train_reduced)
cls2_all.score(x_test_red_st, y_test_reduced)
#0.9521531100478469


# # 7. Model Selection

# ## 7.1. Selecting Best Models Using Exhaustive Search

# In[75]:


from sklearn.model_selection import GridSearchCV

# Create RandomForestClassifier
classifier = RandomForestClassifier()

# Create hyperparameter options
grid_param = {
    'n_estimators': [100, 150, 200 ,300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


gd_sr = GridSearchCV(estimator=classifier, param_grid=grid_param, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)


# In[76]:


gd_sr.fit(features_train, target_train)


# In[77]:


best_parameters = gd_sr.best_params_
print(best_parameters)


# ## 7.2. Evaluating Performance After Model Selection

# In[78]:


best_result = gd_sr.best_score_
print(best_result)
#0.9521531100478469


# In[ ]:




