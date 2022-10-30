# Encoding

- check category features

```py
# Get data that have object dtype 
categorical_features = [c for c in df if df[c].dtype=='O']
categorical_features
```

- ordinal

```py
encoding_data = {"false": 0, "true":1}
df.replace({"col1": encoding_data}, inplace=True)
```

- binary

```py
import category_encoders

binary_encoder = category_encoders.BinaryEncoder()

category_features = ["col1", "col2", "col3"]
for category_feature in category_features:
    encoder = binary_encoder(cols=category_feature)
    df = encoder.fit_transform(df)
```

- weight of evidence (WOE) and information value (IV)

```py
#Suitable for converting data categories to the form of regression values with the target class is binary classification
def cal_woe_iv(df, feature, target):
  df_woe_iv = (pd.crosstab(df[feature],df[target],
                           normalize='columns')
                            .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                            .assign(iv=lambda dfx: np.sum(dfx['woe']*(dfx[1]-dfx[0]))))
  dict_temp = {}
  for x, y in df_woe_iv.iterrows():
    y_woe = y['woe']
    if(y['woe']==float('inf')):
      y_woe = -11.512925465
    elif(y['woe']==float('-inf')):
      y_woe = -11.512925465
    dict_temp[x] = y_woe
  
  return dict_temp


category_woe = {}
features = ["col1", "col2", "col3", "col4", "col5"]
target = "col_target"
for feature in features:
  category_woe[feature] = cal_woe_iv(df_train, feature, target)

for feature in features:
  df_train.replace({feature: category_woe[feature]}, inplace=True)
```
