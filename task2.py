import pandas as pd
# Sample dataset
data = {
    "Age": [12, 37, 44, 50, 65],
    "Salary": [25000, 35000, 60000, 75000, 110000]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
# standardization
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()

df_standardized = pd.DataFrame(
    scaler_std.fit_transform(df),
    columns=df.columns
)

print(df_standardized)

# normalization
from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler()

df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=df.columns
)

print(df_normalized)
import matplotlib.pyplot as plt

# Before Scaling
plt.figure()
plt.hist(df["Salary"], bins=5)
plt.title("Salary - Before Scaling")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()


# After Standardization
plt.figure()
plt.hist(df_standardized["Salary"], bins=5)
plt.title("Salary - After Standardization")
plt.xlabel("Scaled Salary")
plt.ylabel("Frequency")
plt.show()


# After Normalization
plt.figure()
plt.hist(df_normalized["Salary"], bins=5)
plt.title("Salary - After Normalization")
plt.xlabel("Scaled Salary (0-1)")
plt.ylabel("Frequency")
plt.show()