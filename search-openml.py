import openml
df = openml.datasets.list_datasets(output_format="dataframe")

print(df.columns)

need = (
    (df["NumberOfSymbolicFeatures"].fillna(0) >= 1) &
    (df["NumberOfNumericFeatures"].fillna(0) >= 1)
)

sets = df.loc[need, ["did","name",
              "NumberOfSymbolicFeatures","NumberOfNumericFeatures"]]

print(len(sets))
print(sets.head())