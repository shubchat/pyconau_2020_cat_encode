import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression




def model_and_predict(df_train,df_test,Cat_predictors):
    df_train.drop(Cat_predictors+['Name','Ticket','PassengerId'],axis=1,inplace=True)
    df_test.drop(Cat_predictors+['Name','Ticket','PassengerId'],axis=1,inplace=True)
    predictors=df_train.loc[:,df_train.columns!='Survived'].to_numpy()
    outcome=df_train['Survived'].to_numpy()
    clf = LogisticRegression(solver='liblinear')
    clf.fit(predictors,outcome)
    AUC_train=roc_auc_score(df_train['Survived'],clf.predict_proba(df_train.loc[:,df_train.columns!='Survived'])[:,1])
    AUC_test=roc_auc_score(df_test['Survived'],clf.predict_proba(df_test.loc[:,df_test.columns!='Survived'])[:,1])
    Incremental_above_random=(AUC_test-0.5289873417721519)/0.5289873417721519
    
    print("Training AUC is {},\nTest AUC is {}\nIncremental improvement over random is {}" .format(AUC_train,AUC_test,Incremental_above_random))
          
 
    
   
    