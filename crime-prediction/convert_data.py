import pandas as pd

df = pd.read_csv("data/crime_data.csv")

print("COLUMNS ARE:")
for col in df.columns:
    print(f"'{col}'")