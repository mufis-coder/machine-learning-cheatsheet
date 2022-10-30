# Imputation

- numeric data

```py
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=nan, strategy="mean")
mean_imputer.fit(df["col1"].values.reshape(-1, 1))

df["col1"] = mean_imputer.transform(df["col1"].values.reshape(-1, 1))
```

```py
median_imputer = SimpleImputer(missing_values=nan, strategy="median")
median_imputer.fit(df["col1"].values.reshape(-1, 1))

df["col1"] = median_imputer.transform(df["col1"].values.reshape(-1, 1))
```

- category data

```py
for feature in category_features:
    df["feature"].fillna(df["feature"].mode()[0], inplace=True)
```
