import pandas as pd

#1) Dummies replacement

#drop_first set to true to avoid the dummy trap
var = pd.get_dummies(df["var1"], drop_first = True)
