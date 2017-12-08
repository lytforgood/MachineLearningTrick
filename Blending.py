# -*- coding: utf-8 -*-
from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#加载数据集
from sklearn.datasets import load_boston
data = load_boston()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=111)
#创建数据集
dataset = Dataset(X_train,y_train,X_test)
#创建RF模型和LR模型
model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50},name='rf')
model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True},name='lr')
# Blending两个模型
# Returns new dataset with out-of-fold predictions
pipeline = ModelsPipeline(model_rf,model_lr)
stack_ds = pipeline.blend(proportion=0.2,seed=111)
#第二层使用lr模型stack
stacker = Regressor(dataset=stack_ds, estimator=LinearRegression)
results = stacker.predict()
# 使用10折交叉验证结果
results10 = stacker.validate(k=10,scorer=mean_absolute_error)
