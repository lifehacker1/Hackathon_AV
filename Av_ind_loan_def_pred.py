import pandas as pd
from time import strptime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



train_p=r"C:\Users\new\Desktop\av_loan_def_pred\train_u5jK80M\train.csv"
test_p=r"C:\Users\new\Desktop\av_loan_def_pred\test_3BA6GZX\test.csv"
train_df= pd.read_csv(train_p)
test_df=pd.read_csv(test_p)


#feature eng

##null check

train_df.isnull().any()
train_df.isnull().sum()

##drop unwanted 

columns=[col for col in train_df.columns ] 

def dropUnwanted(df,columns):
    df.drop(columns,axsis=1)
    
 
def getDummies(df,columns):
    pd.get_dummies(df,columns=columns)
    
drop_columns=[columns[0]]



##########

dummy_columns=["source","financial_institution","loan_purpose"]

train_df=pd.get_dummies(train_df,columns=dummy_columns)
test_df=pd.get_dummies(test_df,columns=dummy_columns)



train_df["first_payment"]=train_df["first_payment_date"].apply(lambda x:int(x.split("/")[0]) )-train_df["origination_date"].apply(lambda x:int(x.split("-")[1]))
test_df["first_payment"]=test_df["first_payment_date"].apply(lambda x:strptime(x.split("-")[0],"%b").tm_mon)-test_df["origination_date"].apply(lambda x:int(x.split("/")[1]))


train_df["max_m"]=train_df.iloc[:, 13:25].max(axis=1)
test_df["max_m"]=test_df.iloc[:, 13:25].max(axis=1)

test_df["sum_m"]=test_df.iloc[:, 13:25].sum(axis=1)
train_df["sum_m"]=train_df.iloc[:, 13:25].sum(axis=1)


x_train=train_df.drop(columns=["loan_id","m13","first_payment_date","origination_date"],axis=1)
y_train=train_df["m13"]
x_test=test_df.drop(["loan_id","first_payment_date","origination_date"],axis=1)




############## models walk ############# 


### logistic regression
from sklearn.linear_model import LogisticRegression
import sklearn
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_train)

df_table = confusion_matrix(y_train,y_pred)
print (df_table)
p = df_table[1,1] / (df_table[1,1] + df_table[0,1])
r = df_table[1,1] / (df_table[1,1] + df_table[1,0])


f1_score=sklearn.metrics.f1_score(y_train,y_pred)
accuracy=sklearn.metrics.accuracy_score(y_train,y_pred)




## random forest 
from sklearn.ensemble import RandomForestClassifier
import sklearn
rfm=RandomForestClassifier(n_estimators=200,n_jobs=-1,oob_score=True,random_state=101,max_features=None ,min_samples_leaf=30)



import timeit
start_time = timeit.default_timer()
rfm.fit(x_train,y_train)
elapsed = timeit.default_timer() - start_time




y_pred_rfm=rfm.predict(x_train)
randomforest_f1_200=sklearn.metrics.f1_score(y_train,y_pred_rfm)
randomforest_accuracy_200=sklearn.metrics.accuracy_score(y_train,y_pred_rfm)


import pandas as pd
feature_importances = pd.DataFrame(rfm.feature_importances_,index = x_train.columns, columns=['importance']).sort_values('importance',ascending=False)

r=rfm.predict(x_test)
l=test_df["loan_id"]
results=pd.DataFrame({"loan_id":l,"m13":r})
results.to_csv(r"D:results.csv",index=False)




########## xgboost 

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4



xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

xgb1.fit(x_train, y_train,eval_metric='auc')


y_pred_xgb1=xgb1.predict(x_train)
import sklearn
f1_score_xgb1_200=sklearn.metrics.f1_score(y_train,y_pred_xgb1)
accuracy=sklearn.metrics.accuracy_score(y_train,y_pred_xgb1)

##feature importance
feature_importances = pd.DataFrame(xgb1.feature_importances_,index = x_train.columns, columns=['importance']).sort_values('importance',ascending=False)

pt=feature_importances.plot.bar


r=xgb1.predict(x_test)
l=test_df["loan_id"]
results=pd.DataFrame({"loan_id":l,"m13":r})


results.to_csv(r"D:results_xgb1_200.csv",index=False)


feat_imp = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')



print(xgb1.feature_importances_)
# plot
from matplotlib import pyplot
pyplot.bar(range(len(xgb1.feature_importances_)), xgb1.feature_importances_)
pyplot.show()


from xgboost import plot_importance
plot_importance(xgb1)
pyplot.show()

















