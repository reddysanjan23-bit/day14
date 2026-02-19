import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample data
df = pd.DataFrame({
    "Transmission": ["Automatic", "Manual", "Automatic", "Manual"],
    "Color": ["Red", "Blue", "Yellow", "OrangeS"]
})

# 1. Label Encoding - Transmission
le = LabelEncoder()
df["Transmission_Encoded"] = le.fit_transform(df["Transmission"])

# 2. One-Hot Encoding - Color
df = pd.get_dummies(df, columns=["Color"], drop_first=True)

print(df)