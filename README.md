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

df["col1"].value_counts()
```

- select features

```py
#delete columns
df.drop(columns=["col1", "col2", "col3"], inplace=True)
#select columns
columns = ["col1", "col2", "col3"]
df = df[columns]
#delete row that contain nan value
df = df.dropna()
```

- dates

```py
#format Year-Month-Day Hour:Minute:Second
df["col_date"] = pd.to_datetime(df['col_date_before'], format="%Y-%m-%d %H:%M:%S")
#date difference in days
dif = (date1-date2).dt.days
```

[Category to numerik (encoding)]({link})
