# Hyperparameter Tuning

## Xgboost Classifier

- learning_rate: Learning rate shrinks the weights to make the boosting process more conservative
- max_depth: Maximum depth of the tree, increasing it increases the model complexity.
- gamma: Gamma specifies the minimum loss reduction required to make a split.
- colsample_bytree: Percentage of columns to be randomly samples for each tree.
- reg_alpha: reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
- reg_lambda: reg_lambda provides l2 regularization to the weight, higher values result in more conservative models

```py
!pip install xgboost
import xgboost as xgb

param_grid = {"learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
              "max_depth": range(3, 21, 3),
              "gamma": [i/10.0 for i in range(0, 5)],
              "colsample_bytree": [i/10.0 for i in range(3, 10)],
              "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
              "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]}

model = xgb.XGBClassifier()
```

## Regression

```py
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, Lars
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
```

## LinearRegression

```py
param_grid = {
    "fit_intercept": [True, False],
    "positive": [False, True],
}
model = LinearRegression()
```

## ElasticNet

```py
param_grid = {
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
    'l1_ratio': [i/20 for i in range(20+1)]
}
model = ElasticNet()
```

## DecisionTreeRegressor

```py
param_grid = {
    "criterion": ['squared_error', 'friedman_mse'],
    "max_depth": [2, 4, 8, 12, 16, 20, None],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
}
model = DecisionTreeRegressor()
```

## RandomForestRegressor

```py
param_grid = {'bootstrap': [True, False],
              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
model = RandomForestRegressor()
```

## Xgboost Regressor

```py
param = {'max_depth': [i for i in range(3, 10)],
         'booster': ['gblinear', 'dart', 'tree'],
         'min_child_weight': [i for i in range(0, 10, 2)],
         'n_estimators': [180]
         }
model = xgb.XGBRFRegressor()
```

## CatBoostRegressor

```py
param_grid = {
    'depth': [5, 10, 50, 100],
    'learning_rate': [0.005, 0.01, 0.05],
    'iterations': [10, 50, 100, 150]}
model = cb.CatBoostRegressor(loss_function='MultiRMSE', verbose=0)
```

scoring can be accessed in: [HERE!](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

## RandomizedSearchCV

```py
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

scoring = ['recall'] 
# Set up the k-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

random_search = RandomizedSearchCV(estimator=xgboost, 
                           param_distributions=param_grid, 
                           n_iter=50,
                           scoring=scoring, 
                           refit='recall', 
                           n_jobs=-1, 
                           cv=kfold, 
                           verbose=0)

random_result = random_search.fit(X_train, y_train)

# Print the best score and the corresponding hyperparameters
print(f'The best score is {random_result.best_score_:.4f}')
print(f'The best hyperparameters are {random_result.best_params_}')

# Make prediction using the best model
random_predict = random_search.predict(X_test)
```

## GridSearchCV

```py
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=[
                           'neg_root_mean_squared_error'], refit='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_search.get_params()
grid_search.best_score_
grid_search.best_estimator_.get_params()
```
