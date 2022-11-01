# Multiple Models

## Model Classfiers

```py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

model_clas = [
    ('LogisticRegression', LogisticRegression()),
    ('DecisionTreeClassifier', DecisionTreeClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('XGBoostClassifier', xgb.XGBClassifier()),
    ('CatBoostRegressor', CatBoostClassifier()),
    ]
```

- evaluation classification

```py
from sklearn.metrics import f1_score, confusion_matrix

def calc_f1score(y_true, y_pred):
    score = f1_score(y_true, y_pred)*100
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f'F1:{score}')
    print("confusion matrix")
    print(cf_matrix)
    return score
```

## Model Regression

```py
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


models_reg = [('LinearRegression', MultiOutputRegressor(LinearRegression())),
                ('SGDRegressor', MultiOutputRegressor(SGDRegressor())),
                ('Lars', MultiOutputRegressor(Lars(normalize=False))),
                ('Decision Tree', MultiOutputRegressor(DecisionTreeRegressor())),
                ('SVM', MultiOutputRegressor(SVR())),
                ('KNNRegressor', MultiOutputRegressor(KNeighborsRegressor())),
                ('Elastic-Net', MultiOutputRegressor(ElasticNet())),
                ('MLPRegressor', MLPRegressor(activation='tanh', learning_rate_init=0.005, solver='sgd')),

                ('LightGBM',  lgb.LGBMRegressor()),
                ('RandomForestRegressor', RandomForestRegressor()),
                ('XGBoostRegressor', xgb.XGBRFRegressor()),
                ('CatBoostRegressor', cb.CatBoostRegressor(loss_function='MultiRMSE', verbose=0))]
```

- evaluation regression

```py
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calc_mse_mae_rmse(y_true, y_pred):
    mses = mean_squared_error(
        y_true, y_pred, multioutput='raw_values', squared=True)
    rmses = mean_squared_error(
        y_true, y_pred, multioutput='raw_values', squared=False)
    maes = mean_absolute_error(y_true, y_pred, multioutput='raw_values')

    print(f'MSE:{mses}')
    print(f'RMSE:{rmses}')
    print(f'MAE:{maes}')

    return mses, rmses, maes
```

## Training

```py
import time
def train_models(models, x, y):
    models_trained = {"model_name": [], "model": []}
    for name, model in models:
        print('training:', name, end='|')
        start_time = time.perf_counter()
        model.fit(x, y)
        end_time = time.perf_counter()
        print(f'That took: {(end_time - start_time)}')

        models_trained['model_name'].append(name)
        models_trained['model'].append(model)
    return models_trained

models_trained = train_models(models, X_train, y_train)
```

## Evaluation

```py
def calc_score(models_trained, x, y):
  Metrics = {"model_name":[], "score":[]}
  for i in range(0, 5):
    model = models_trained['model'][i]
    name = models_trained['model_name'][i]

    print('predicting by:', name)
    y_pred = model.predict(x)
    score = calc_f1score(y, y_pred) || calc_mse_mae_rmse(y, y_pred)

    Metrics["model_name"].append(name)
    Metrics["f1_score"].append(score)
  return Metrics

scores = calc_score(models_trained, X_test, y_test)
scores_df = pd.DataFrame.from_dict(met_dict)
```

## Saving Model

```py
from joblib import dump
for name, model in models_trained:
  filePath = f'{basePath}/{name}-{description-model}.sav'
  dump(model, filePath)
```
