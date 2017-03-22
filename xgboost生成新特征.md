##导入模块使用
需要根据实际情况修改xgboost参数
```
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
from Xgboost_Feature import XgboostFeature

X, y = make_hastie_10_2(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)##test_size测试集合所占比例
##自己设置xgboost模型参数 默认树个数30 
model=XgboostFeature(n_estimators=50)
##切分训练集训练叶子特征模型 返回值是 原特征+新特征
X_train,y_train, X_test, y_test=model.fit_model_split(X_train, y_train,X_test, y_test)
##不切分训练集训练叶子特征模型  返回值 是原特征+新特征
X_train,y_train, X_test, y_test=model.fit_model(X_train, y_train,X_test, y_test)
```
##详细过程
```python
from sklearn.model_selection import train_test_split
from sklearn import metrics
from  sklearn.datasets  import  make_hastie_10_2
from xgboost.sklearn import XGBClassifier
##载入示例数据 10维度
X, y = make_hastie_10_2(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)##test_size测试集合所占比例
##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)
```


```python
##合并维度
import numpy as np
def mergeToOne(X,X2):
    X3=[]
    for i in xrange(X.shape[0]):
        tmp=np.array([list(X[i]),list(X2[i])])
        X3.append(list(np.hstack(tmp)))
    X3=np.array(X3)
    return X3
```


```python
clf = XGBClassifier(
 learning_rate =0.3, #默认0.3
 n_estimators=30, #树的个数
 max_depth=3,
 min_child_weight=1,
 gamma=0.5,
 subsample=0.6,
 colsample_bytree=0.6,
 objective= 'binary:logistic', #逻辑回归损失函数
 nthread=4,  #cpu线程数
 scale_pos_weight=1,
 reg_alpha=1e-05,
 reg_lambda=1,
 seed=27)  #随机种子
```


```python
clf.fit(X_train_1, y_train_1)
new_feature= clf.apply(X_train_2)
```


```python
X_train_new2=mergeToOne(X_train_2,new_feature)
new_feature_test= clf.apply(X_test)
X_test_new=mergeToOne(X_test,new_feature_test)
```


```python
model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.5,
 subsample=0.6,
 colsample_bytree=0.6,
 objective= 'binary:logistic', 
 nthread=4, 
 scale_pos_weight=1,
 reg_alpha=1e-05,
 reg_lambda=1,
 seed=27)
```


```python
model.fit(X_train_new2, y_train_2)
y_pre= model.predict(X_test_new)
y_pro= model.predict_proba(X_test_new)[:,1] 
print "AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro) 
print"Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre) 
```

    AUC Score : 0.993206
    Accuracy : 0.955



```python
model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=2000,
 max_depth=3,
 min_child_weight=1,
 gamma=0.6,
 subsample=0.7,
 colsample_bytree=0.8,
 objective= 'binary:logistic', 
 nthread=4, 
 scale_pos_weight=1,
 seed=27)
model.fit(X_train_new2, y_train_2)
y_pre= model.predict(X_test_new)
y_pro= model.predict_proba(X_test_new)[:,1] 
print "AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro) 
print"Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre) 
```

    AUC Score : 0.993673
    Accuracy : 0.9604



```python

```
