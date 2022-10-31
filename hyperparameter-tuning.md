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
```
