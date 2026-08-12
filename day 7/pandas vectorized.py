import pandas as pd
names=pd.Series(['Alice','bob','CHARLIE'])
print(names.str.lower())
print(names.str.contains('a'))
print(names.str.upper())
print(names.str)