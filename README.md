# Cheatsheet Machine Learning

## Pandas

- read and save file tabular

```py
import pandas as pd

path = "" #string -> the path where the file is located
df = pd.read_csv(path) # read csv file
df = pd.read_excel(path) # read excel file

...

path_out = "" #string -> the path where file output will be saved
df.to_csv(path_out, index=False)
```

- basic info

```py
df.shape

df.info()

df.columns

df["col1"].value_counts()

df.describe()

df.describe().transpose()
```

- select features

```py
#delete columns
df.drop(columns=["col1", "col2", "col3"], inplace=True, axis=1)

#select columns
columns = ["col1", "col2", "col3"]
df = df[columns]

#delete row that contain nan value
df = df.dropna()
```

- select rows

```py
#single condition
df = df[df["cols1"]=="value1"]

#multiple conditions
df = df[df['cols1'].isin(["value1", "value2", "value3"])]
```

- Concat df vertically

```py
df_merged = df1.append([df2, df3], ignore_index=True)
```

- Add columns with value from list

```py
df1['col_new'] = [value1, value2, value3]
```

- dates

```py
#format Year-Month-Day Hour:Minute:Second
df["col_date"] = pd.to_datetime(df['col_date_before'], format="%Y-%m-%d %H:%M:%S")

#date difference in days
dif = (date1-date2).dt.days
```

- null values

```py
df.isnull().sum()

df.isnull().sum() / df.shape[0] * 100.00

df = df.dropna()
```

- make dataframe using dictionary
  
```py
temp_dict = {"cols1":[], "cols2":[]}

for index, row in df.iterrows():
    temp_dict["cols1"].append(row["cols_df1"])
    temp_dict["cols2"].append(row["cols_df2"])

df_new = pd.DataFrame(temp_dict)
```

- print all str in dataframe

```py
df_train.review_text.values.astype(str)
```

- [Fill nan value (imputation)](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/imputation.md)

- [Normalization](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/normalization.md)

- [Category to numerik (encoding)](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/encoding.md)

- [Split and Balance data](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/split-balancing-data.md)

- [Train multiple models](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/multiple-models.md)

- [Hyperparameter Tuning](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/hyperparameter-tuning.md)

- [Visualization](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/visualization.md)

- [Thread, batch, and tqdm](https://github.com/mufis-coder/machine-learning-cheatsheet/blob/main/thread-batch-tqdm.md)
