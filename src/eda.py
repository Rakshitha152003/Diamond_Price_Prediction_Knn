# 1. Import Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# 2. Set Project Directory

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diamonds.csv"


# 3. Load Dataset

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED SUCCESSFULLY ==========\n")


# 4. Dataset Dimensions

print("Dataset Shape:", df.shape)

print("Number of Rows:", df.shape[0])

print("Number of Columns:", df.shape[1])


# 5. Display First Five Records

print("\n========== FIRST FIVE RECORDS ==========\n")

print(df.head())


# 6. Display Last Five Records

print("\n========== LAST FIVE RECORDS ==========\n")

print(df.tail())


# 7. Dataset Information

print("\n========== DATASET INFORMATION ==========\n")

df.info()


# 8. Statistical Description

print("\n========== STATISTICAL SUMMARY ==========\n")

print(df.describe())


# 9. Check Missing Values

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())


# 10. Check Duplicate Records

print("\n========== DUPLICATE RECORDS ==========\n")

print("Total Duplicates:", df.duplicated().sum())


# 11. Check Data Types

print("\n========== DATA TYPES ==========\n")

print(df.dtypes)


# 12. Check Unique Values

print("\n========== UNIQUE VALUES ==========\n")

for column in df.columns:
    print(column, ":", df[column].nunique())


# 13. Target Variable Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df['price'], kde=True)

plt.title("Distribution of Diamond Prices")

plt.xlabel("Diamond Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()


# 14. Categorical Feature Distribution

categorical_features = ['cut', 'color', 'clarity']

for column in categorical_features:

    plt.figure(figsize=(8, 5))

    sns.countplot(data=df, x=column)

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)

    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()


# 15. Numerical Feature Distribution

numerical_features = [
    'carat', 'depth', 'table', 'x', 'y', 'z'
]

for column in numerical_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(df[column], kde=True)

    plt.title(f"Distribution of {column}")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


# 16. Outlier Detection Using Boxplots

for column in numerical_features:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(f"Outlier Detection - {column}")

    plt.tight_layout()

    plt.show()


# 17. Relationship Between Carat and Price

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x='carat', y='price')

plt.title("Carat vs Diamond Price")

plt.xlabel("Carat")

plt.ylabel("Price")

plt.tight_layout()

plt.show()


# 18. Correlation Analysis

plt.figure(figsize=(10, 7))

correlation = df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()
