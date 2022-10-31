# Model Classfiers

```py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

models = [
    ('LogisticRegression', LogisticRegression()),
    ('DecisionTreeClassifier', DecisionTreeClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('XGBoostClassifier', xgb.XGBClassifier()),
    ('CatBoostRegressor', CatBoostClassifier()),
    ]

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

- evaluation

```py
from sklearn.metrics import f1_score, confusion_matrix

def calc_f1score(name, y_true, y_pred):
    score = f1_score(y_true, y_pred)*100
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f'F1:{score}')
    print("confusion matrix")
    print(cf_matrix)
    return score

def calc_score(models_trained, x, y):
  Metrics = {"model_name":[], "score":[]}
  for i in range(0, 5):
    model = models_trained['model'][i]
    name = models_trained['model_name'][i]

    print('predicting by:', name)
    y_pred = model.predict(x)
    score = calc_f1score(name, y, y_pred)

    Metrics["model_name"].append(name)
    Metrics["f1_score"].append(score)
  return Metrics

scores = calc_score(X_test, y_test)
scores_df = pd.DataFrame.from_dict(met_dict)
```
