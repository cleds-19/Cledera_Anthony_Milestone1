import pandas as pd
import matplotlib.pyplot as plt

# Load the raw dataset
file_name = "Fundamentals_of_Data_Mining_Milestone1_Raw_Dataset.csv"
df = pd.read_csv(file_name)

print("=" * 60)
print("FUNDAMENTALS OF DATA MINING - MILESTONE 1")
print("CLEAN, TRANSFORM, AND EXPLORE")
print("=" * 60)

# ============================================================
# A. DATA LOADING
# ============================================================
print("\nA. DATA LOADING")

print("\nFirst five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

# ============================================================
# B. DATA INSPECTION
# ============================================================
print("\nB. DATA INSPECTION")

print("\nNumber of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nBasic statistical summary:")
print(df.describe())

# ============================================================
# C. DATA CLEANING
# ============================================================
print("\nC. DATA CLEANING")

# Display the rows containing missing values
print("\nRows with missing values:")
print(df[df.isnull().any(axis=1)])

# Remove duplicate records
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

# Fill missing numerical values using the median
numeric_columns = [
    "Age",
    "Attendance",
    "Quiz_Score",
    "Assignment_Score"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Check for invalid values outside the valid 0-100 range
print("\nInvalid numerical values before correction:")

for column in ["Attendance", "Quiz_Score", "Assignment_Score"]:
    invalid = df[~df[column].between(0, 100)]
    print(f"\n{column}:")
    print(invalid[["Student_ID", column]])

# Replace invalid values with the median of valid values
for column in ["Attendance", "Quiz_Score", "Assignment_Score"]:
    valid_median = df.loc[df[column].between(0, 100), column].median()
    df.loc[~df[column].between(0, 100), column] = valid_median

# Standardize text values
df["Gender"] = df["Gender"].str.strip().str.title()
df["Course"] = df["Course"].str.strip().str.upper()

# Verify the cleaned dataset
print("\nCleaning verification:")
print("Missing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate records after cleaning:")
print(df.duplicated().sum())

# ============================================================
# D. DATA TRANSFORMATION
# ============================================================
print("\nD. DATA TRANSFORMATION")

# Average_Score uses the quiz and assignment scores
df["Average_Score"] = (
    df["Quiz_Score"] + df["Assignment_Score"]
) / 2

# Create Score_Category
def classify_score(score):
    if 90 <= score <= 100:
        return "Excellent"
    elif 80 <= score < 90:
        return "Good"
    else:
        return "Needs Improvement"

df["Score_Category"] = df["Average_Score"].apply(classify_score)

print("\nTransformed dataset:")
print(df.head())

# ============================================================
# E. BASIC EXPLORATORY DATA ANALYSIS
# ============================================================
print("\nE. BASIC EXPLORATORY DATA ANALYSIS")

overall_average = df["Average_Score"].mean()
highest_average = df["Average_Score"].max()
lowest_average = df["Average_Score"].min()

highest_student = df.loc[
    df["Average_Score"].idxmax(), "Student_ID"
]

lowest_student = df.loc[
    df["Average_Score"].idxmin(), "Student_ID"
]

category_counts = df["Score_Category"].value_counts()

course_average = (
    df.groupby("Course")["Average_Score"]
    .mean()
    .sort_values(ascending=False)
)

highest_course = course_average.index[0]

print(f"\nOverall average score: {overall_average:.2f}")

print(
    f"Highest average score: {highest_average:.2f} "
    f"(Student ID: {highest_student})"
)

print(
    f"Lowest average score: {lowest_average:.2f} "
    f"(Student ID: {lowest_student})"
)

print("\nNumber of students in each Score_Category:")
print(category_counts)

print("\nAverage score by course:")
print(course_average.round(2))

print(
    f"\nCourse with the highest average score: "
    f"{highest_course}"
)

# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(7, 5))

course_average.plot(kind="bar")

plt.title("Average Score by Course")
plt.xlabel("Course")
plt.ylabel("Average Score")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "screenshots/data_analysis.png",
    dpi=150
)

plt.show()

# ============================================================
# SAVE FINAL DATASET
# ============================================================
df.to_csv("cleaned_dataset.csv", index=False)

print("\nFinal cleaned dataset saved as cleaned_dataset.csv")

# ============================================================
# REFLECTION
# ============================================================
print("\nReflection:")
print(
    "Data preprocessing is important because raw data often contains "
    "missing, duplicate, inconsistent, or invalid values."
)
print(
    "Cleaning the data improves its quality and makes the results "
    "more reliable."
)
print(
    "Data transformation converts raw values into useful information "
    "for analysis."
)
print(
    "A clean dataset helps data mining produce more accurate and "
    "meaningful findings."
)
