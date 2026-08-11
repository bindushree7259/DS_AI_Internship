import pandas as pd
x={"math":80,"science":85,"english":90}
y=pd.Series(x)
print(y[y>80])
print (y.iloc[2])
print(y.index[2])
print(y.index[2],":",y.iloc[2])