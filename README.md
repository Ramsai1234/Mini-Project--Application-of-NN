# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:

Smoker Prediction with Medical Costs data 

## Project Description 
An adult who has smoked 100 cigarettes in his or her lifetime and who currently smokes cigarettes. Beginning in 1991 this group was divided into “everyday” smokers or “somedays” smokers. Environmental Tobacco Smoke (ETS): Also called second-hand smoke. Inhaling ETS is called passive smoking.



## Algorithm:

1.import the necessary pakages.

2.install the csv file

3.using the for loop and predict the output

4.plot the graph

5.analyze the regression bar plot




## Program:
```
Developed by:P.Ramsai
reg.no:212221240041


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler



from sklearn.metrics import accuracy_score, classification_report, plot_roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('insurance.csv')
df.head()

df.info()

le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

dict = {}
for i in list(df.columns):
    dict[i] = df[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()

fig = plt.figure(figsize=(18,15))
gs = fig.add_gridspec(2,3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

sns.barplot(ax=ax0,data=df,x='charges',y = 'age')
sns.barplot(ax=ax1,data=df,x='sex',y = 'charges')
sns.barplot(ax=ax2,data=df,x='charges',y = 'bmi')
sns.barplot(ax=ax3,data=df,x='children',y = 'charges')
sns.barplot(ax=ax4,data=df,x='smoker',y = 'charges')
sns.barplot(ax=ax5,data=df,x='region',y = 'charges')

df.corr()

plt.matshow(df.corr())
plt.show()

corr = df.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_precision(2)
    
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


y, X = dmatrices('charges ~ age+bmi+sex+children+smoker+region+charges', data=df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns


vif

x_train, x_test, y_train, y_test = train_test_split(df.drop('smoker', axis = 1),
                                                   df['smoker'],
                                                   test_size = 0.25,
                                                   random_state=42)
                                                   
 pipe1 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(random_state=42))
])

model_svm = pipe1.fit(x_train, y_train)

y_pred_svm = model_svm.predict(x_test)

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

model_kn = pipe2.fit(x_train, y_train)

y_pred_kn = model_kn.predict(x_test)

pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

model_lr = pipe3.fit(x_train, y_train)

y_pred_lr = model_lr.predict(x_test)

model_rf = RandomForestClassifier(random_state=42).fit(x_train, y_train)

y_pred_rf = model_rf.predict(x_test)
model_gb = GradientBoostingClassifier(random_state=42).fit(x_train, y_train)

y_pred_gb = model_gb.predict(x_test)
model_dt = DecisionTreeClassifier(random_state=42).fit(x_train, y_train)

y_pred_dt = model_dt.predict(x_test)

print('Accuracy score SVM: {:.4f}' .format(accuracy_score(y_test, y_pred_svm)))
print('Accuracy score KN:  {:.4f}' .format(accuracy_score(y_test, y_pred_kn)))
print('Accuracy score LR:  {:.4f}' .format(accuracy_score(y_test, y_pred_lr)))
print('Accuracy score RF:  {:.4f}' .format(accuracy_score(y_test, y_pred_rf)))
print('Accuracy score GB:  {:.4f}' .format(accuracy_score(y_test, y_pred_gb)))
print('Accuracy score DT:  {:.4f}' .format(accuracy_score(y_test, y_pred_dt)))

feat_importances = pd.Series(model_rf.feature_importances_, index = x_train.columns).sort_values(ascending = True)
feat_importances.plot(kind = 'barh')
plt.title('Feature Importances ')
plt.show()

```
## Output:



![Screenshot 2022-12-04 230655](https://user-images.githubusercontent.com/94269989/205506566-3687c725-c0d7-4bf5-8a4e-eb313790c7cb.png)



![Screenshot 2022-12-04 230711](https://user-images.githubusercontent.com/94269989/205506576-e35ac38e-f319-42f6-abd4-0dded77a2e5f.png)


![Screenshot 2022-12-04 230734](https://user-images.githubusercontent.com/94269989/205506680-63ada54d-ed9c-45cb-b624-225a69afbba4.png)


![Screenshot 2022-12-04 230936](https://user-images.githubusercontent.com/94269989/205506685-ae3041fa-d19e-46f1-82af-bf4b47ce747c.png)


![Screenshot 2022-12-04 231011](https://user-images.githubusercontent.com/94269989/205506692-bd5e28fd-0222-421a-ac40-8b7fcef831e4.png)


![Screenshot 2022-12-04 231040](https://user-images.githubusercontent.com/94269989/205506696-f71c64f9-7c52-4ae6-984f-5cd70aca6856.png)

![Screenshot 2022-12-04 231102](https://user-images.githubusercontent.com/94269989/205506700-92ebb54b-9e14-48c1-8c11-02d05abdcdcf.png)


![Screenshot 2022-12-04 231124](https://user-images.githubusercontent.com/94269989/205506703-6cbb1636-5015-45e6-9775-7c0780d96d7b.png)

## Advantage :

1.This report (The Health Benefits of Smoking Cessation) describes the associations between smoking cessation and changes in risk for specific disease outcomes. 

2.It also addresses how cessation affects the natural history of various disease outcomes.

3.such as by slowing the progression of underlying pathophysiological processes.

## Result:
Thus the implementing of smoker prediction with medical cost is sucessfully done by using python code.
