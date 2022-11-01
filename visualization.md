# Visualization

## Heatmap Correlation

```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 10))

# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
```

## Subplot

```py
plt.figure(1 , figsize = (21, 15))

plt.subplot(2 , 3 , 1)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df1, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title1")
plt.ylim(0.0, 1.0)

plt.subplot(2 , 3 , 2)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df2, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title2")
plt.ylim(0.0, 1.0)

plt.subplot(2 , 3 , 3)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df3, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title3")
plt.ylim(0.0, 2.0)

plt.subplot(2 , 3 , 4)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df4, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title4")
plt.ylim(0.0, 1.0)

plt.subplot(2, 3 , 5)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df5, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title5")
plt.ylim(0.0, 1.0)

plt.subplot(2, 3 , 6)
plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
sns.barplot(data=df6, x="col_x", y="col_y", hue="hue_data", ci=None)
plt.title("title6")
plt.ylim(0.0, 2.0)

plt.show()
```

- subplot using forloop

```py
figure, axis = plt.subplots(4, 3, figsize=(32,24))
for i,colname in enumerate(df.columns.values):
  row = i//3
  col = i%3
  axis[row, col].scatter(df[colname], df['ACCELEROMETER X (m/sÂ²)'])
  axis[row, col].set_title(f'ACCELEROMETER X (m/sÂ²) & {colname}')
  print("finish")
```
