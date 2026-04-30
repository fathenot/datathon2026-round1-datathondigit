import pandas as pd
import matplotlib.pyplot as plt
import pickle

# load submission data
submission = pd.read_csv("./submission.csv")
submission["Date"] = pd.to_datetime(submission["Date"])
# extract year + month
submission["year"] = submission["Date"].dt.year
submission["month"] = submission["Date"].dt.month

# aggregate theo tháng
monthly = submission.groupby(["year", "month"])["Revenue"].sum().reset_index()
submission = monthly[monthly["year"] == 2024]

newest_year = pd.read_csv("../sales.csv")
newest_year["Date"] = pd.to_datetime(newest_year["Date"])
newest_year["year"] = newest_year["Date"].dt.year
newest_year["month"] = newest_year["Date"].dt.month
newest_year = newest_year.groupby(["year", "month"])["Revenue"].sum().reset_index()
newest_year = newest_year[newest_year["year"] == 2022]

# plot
plt.figure(figsize=(12,6))

plt.plot(submission["month"], submission["Revenue"], marker='o')
plt.plot(newest_year["month"], newest_year["Revenue"], marker='o')


plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.title("Seasonality Comparison Between Two Years")
plt.legend(["2024", "2022"])
plt.grid()

plt.show()