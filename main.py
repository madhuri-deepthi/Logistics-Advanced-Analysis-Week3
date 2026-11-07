import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GRAPH_DIR = BASE_DIR / "graphs"
REPORT_DIR = BASE_DIR / "reports"

# Create folders
GRAPH_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_DIR / "logistics_dataset.csv")

print("\n===== LOGISTICS ADVANCED DATA ANALYSIS =====")

# Dataset information
print("\nDataset Shape:", df.shape)
print("\nDataset Columns:", df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Visualization style
sns.set_style("whitegrid")

# Graph 1: Delivery Time Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Delivery_Time_Hours"], kde=True)
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Hours)")
plt.ylabel("Number of Shipments")
plt.savefig(GRAPH_DIR / "delivery_time_distribution.png")
plt.close()

# Graph 2: Transportation Cost Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Transportation_Cost"], kde=True)
plt.title("Transportation Cost Distribution")
plt.xlabel("Transportation Cost")
plt.savefig(GRAPH_DIR / "transportation_cost_distribution.png")
plt.close()

# Graph 3: Traffic vs Delivery Time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Traffic_Level", y="Delivery_Time_Hours")
plt.title("Traffic Level vs Delivery Time")
plt.savefig(GRAPH_DIR / "traffic_delivery_time.png")
plt.close()

# Graph 4: Weather vs Delivery Time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Weather_Condition", y="Delivery_Time_Hours")
plt.title("Weather vs Delivery Time")
plt.savefig(GRAPH_DIR / "weather_delivery_time.png")
plt.close()

# Graph 5: Delivery Status
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Delivery_Status")
plt.title("Delivery Status Distribution")
plt.savefig(GRAPH_DIR / "delivery_status_count.png")
plt.close()

# Graph 6: Correlation Heatmap
plt.figure(figsize=(9, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Logistics Data Correlation")
plt.savefig(GRAPH_DIR / "correlation_heatmap.png")
plt.close()

print("\nAll 6 graphs created successfully!")
# Create analysis summary
avg_delivery = df["Delivery_Time_Hours"].mean()
avg_cost = df["Transportation_Cost"].mean()
summary = pd.DataFrame({
    "Metric": [
        "Total Shipments",
        "Total Columns",
        "Missing Values",
        "Duplicate Records",
        "Average Delivery Time",
        "Average Transportation Cost"
    ],
    "Value": [
        len(df),
        len(df.columns),
        int(df.isnull().sum().sum()),
        int(df.duplicated().sum()),
        round(avg_delivery, 2),
        round(avg_cost, 2)
    ]
})

# Save CSV report
summary.to_csv(
    REPORT_DIR / "analysis_summary.csv",
    index=False
)

# Save text report
with open(REPORT_DIR / "EDA_Report.txt", "w") as file:
    file.write("LOGISTICS ADVANCED DATA ANALYSIS REPORT\n")
    file.write("=" * 45 + "\n")
    file.write(f"Total Shipments: {len(df)}\n")
    file.write(f"Total Columns: {len(df.columns)}\n")
    file.write(f"Missing Values: {df.isnull().sum().sum()}\n")
    file.write(f"Duplicate Records: {df.duplicated().sum()}\n")
    file.write(f"Average Delivery Time: {avg_delivery:.2f} hours\n")
    file.write(f"Average Transportation Cost: {avg_cost:.2f}\n")

print("\nEDA Report saved successfully!")
print("WEEK 3 PROJECT COMPLETED SUCCESSFULLY!")