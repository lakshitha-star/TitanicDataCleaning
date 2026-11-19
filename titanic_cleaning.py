import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pathlib import Path

# Get the folder where this Python file is located
folder = Path(__file__).resolve().parent

# 1. Load Titanic dataset
data = pd.read_csv(folder / "titanic.csv")

print("\nFIRST 5 ROWS:")
print(data.head())

# 2. Dataset information
print("\nDATASET INFORMATION:")
data.info()

print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# 3. Check missing values
print("\nMISSING VALUES BEFORE CLEANING:")
print(data.isnull().sum())

# 4. Handle missing values

# Replace missing Age values with median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Replace missing Fare values with mean
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

# Replace missing Embarked values with mode
data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

# Remove Cabin because it contains many missing values
data = data.drop(columns=["Cabin"])

print("\nMISSING VALUES AFTER CLEANING:")
print(data.isnull().sum())

# 5. Label Encoding for Sex
label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])

print("\nSEX AFTER LABEL ENCODING:")
print(data["Sex"].head())

# 6. One-Hot Encoding for Embarked
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

encoded = encoder.fit_transform(data[["Embarked"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Embarked"]),
    index=data.index
)

# Replace original Embarked column with encoded columns
data = pd.concat(
    [data.drop(columns=["Embarked"]), encoded_df],
    axis=1
)

print("\nDATA AFTER ENCODING:")
print(data.head())

# 7. Visualize Age Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    data["Age"],
    bins=20,
    edgecolor="black"
)

plt.title("Titanic Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

# Save graph
graph_path = folder / "age_distribution.png"
plt.savefig(graph_path)

# 8. Save cleaned dataset
csv_path = folder / "cleaned_titanic.csv"
data.to_csv(csv_path, index=False)

# 9. Confirm saved files
print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Cleaned dataset saved at:", csv_path)
print("Age graph saved at:", graph_path)
print("CSV file exists:", csv_path.exists())
print("Graph file exists:", graph_path.exists())

# Display graph after saving both files
plt.show()