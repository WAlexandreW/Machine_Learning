import pandas as pd

#1) Dummies replacement

#drop_first set to true to avoid the dummy trap
var = pd.get_dummies(df["var1"], drop_first = True)


#2) Scaling: either MinMaxscaler or StandargScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = scaler.fit_transform(df)
