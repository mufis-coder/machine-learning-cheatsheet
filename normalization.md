# Normalization

## Fitting

```py
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

sc = StandardScaler()
sc.fit(df)

pathScaller = "" #-> string to path model standarscaler
#save
dump(sc, pathScaller, compress=False)
```

## Transforming

```py
cols = df2.columns
sc = load(pathScaller)
df_scaled = sc.transform(df2)

df_scaled = pd.DataFrame(df_scaled, columns=cols)
```
