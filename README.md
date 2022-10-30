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
