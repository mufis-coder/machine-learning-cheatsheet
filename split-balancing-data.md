# Split and balancing data

- train and test split

```py
from sklearn.model_selection import train_test_split

X = df_train.drop(['feature_target'], axis=1)
Y = df_train['feature_target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# reset index in dataframe
train = train.reset_index(drop=True)
```

- oversampling

```py
from imblearn.over_sampling import SMOTE

oversample = SMOTE(random_state = 43)
X_train_rus, y_train_rus = oversample.fit_resample(X_train, y_train)
```

- undersampling

```py
!pip install -U imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler

undersampling = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus= undersampling.fit_resample(X_train, y_train)
```
