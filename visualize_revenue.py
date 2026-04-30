import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("../sales.csv")
df["Date"] = pd.to_datetime(df["Date"])

# tạo day_of_year
df["day_of_year"] = df["Date"].dt.dayofyear

# tách năm
df1 = df[df["Date"].dt.year == 2021]
df2 = df[df["Date"].dt.year == 2022]

plt.figure(figsize=(14,6))

plt.plot(df1["day_of_year"], df1["Revenue"], label="2021")
plt.plot(df2["day_of_year"], df2["Revenue"], label="2022")

plt.xlabel("Day of Year")
plt.ylabel("Revenue")
plt.title("Revenue Comparison: 2021 vs 2022")
plt.legend()
plt.grid()

plt.show()