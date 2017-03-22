# coding: utf-8
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import numpy as np

class XgboostFeature():
      def __init__(self):
          print('Xgboost Feature start')
      def mergeToOne(self,X,X2):
          X3=[]
          for i in xrange(X.shape[0]):
              tmp=np.array([list(X[i]),list(X2[i])])
              X3.append(list(np.hstack(tmp)))
          X3=np.array(X3)
          return X3
      ##切割训练
      def fit_model_split(self,X_train,y_train,X_test,y_test):
          ##X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合
          X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)
          clf = XGBClassifier(
                 learning_rate =0.3, #默认0.3
                 n_estimators=30, #树的个数
                 max_depth=3,
                 min_child_weight=1,
                 gamma=0.3,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective= 'binary:logistic', #逻辑回归损失函数
                 nthread=4,  #cpu线程数
                 scale_pos_weight=1,
                 reg_alpha=1e-05,
                 reg_lambda=1,
                 seed=27)
          clf.fit(X_train_1, y_train_1)
          y_pre= clf.predict(X_train_2)
          y_pro= clf.predict_proba(X_train_2)[:,1]
          print "pred_leaf=T AUC Score : %f" % metrics.roc_auc_score(y_train_2, y_pro)
          print"pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_train_2, y_pre)
          new_feature= clf.apply(X_train_2)
          X_train_new2=self.mergeToOne(X_train_2,new_feature)
          new_feature_test= clf.apply(X_test)
          X_test_new=self.mergeToOne(X_test,new_feature_test)
          print "Training set of sample size 0.4 fewer than before"
          return X_train_new2,y_train_2,X_test_new,y_test
      ##整体训练
      def fit_model(self,X_train,y_train,X_test,y_test):
          clf = XGBClassifier(
                 learning_rate =0.3, #默认0.3
                 n_estimators=30, #树的个数
                 max_depth=3,
                 min_child_weight=1,
                 gamma=0.3,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective= 'binary:logistic', #逻辑回归损失函数
                 nthread=4,  #cpu线程数
                 scale_pos_weight=1,
                 reg_alpha=1e-05,
                 reg_lambda=1,
                 seed=27)
          clf.fit(X_train, y_train)
          y_pre= clf.predict(X_test)
          y_pro= clf.predict_proba(X_test)[:,1]
          print "pred_leaf=T  AUC Score : %f" % metrics.roc_auc_score(y_test, y_pro)
          print"pred_leaf=T  Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pre)
          new_feature= clf.apply(X_train)
          X_train_new=self.mergeToOne(X_train,new_feature)
          new_feature_test= clf.apply(X_test)
          X_test_new=self.mergeToOne(X_test,new_feature_test)
          print "Training set sample number remains the same"
          return X_train_new,y_train,X_test_new,y_test


