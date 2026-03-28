import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


households = pd.read_csv("households.csv")
timeseries = pd.read_csv("timeseries.csv", parse_dates=["timestamp"])

# Merge metadata for richer analysis
df = timeseries.merge(households, on="household_id", how="left")

# Create extra useful columns
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday_name"] = df["timestamp"].dt.day_name()
df["day_category"] = np.where(df["is_weekend"] == 1, "Weekend", "Weekday")


def save_and_show(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

# 24 hour demand
hourly_demand = df.groupby("hour")["demand"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_demand.index, hourly_demand.values, marker="o")
plt.title("Average Household Demand by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand (kW)")
plt.grid(True, alpha=0.3)
save_and_show("graph_01_avg_demand_by_hour.png")

# avg solar
solar_by_daytype = df.groupby(["day_type", "hour"])["solar"].mean().unstack(0)

plt.figure(figsize=(10, 5))
for col in solar_by_daytype.columns:
    plt.plot(solar_by_daytype.index, solar_by_daytype[col], marker="o", label=col)
plt.title("Average Solar Generation by Hour and Day Type")
plt.xlabel("Hour of Day")
plt.ylabel("Average Solar Generation (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
save_and_show("graph_02_avg_solar_by_daytype.png")

# daily electricity price
hourly_price = df.groupby("hour")["price"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_price.index, hourly_price.values, marker="o")
plt.title("Average Electricity Price by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Price (£/kWh)")
plt.grid(True, alpha=0.3)
save_and_show("graph_03_avg_price_by_hour.png")

# demand vs price scatter
sample_df = df.sample(min(5000, len(df)), random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(sample_df["price"], sample_df["demand"], alpha=0.3)
plt.title("Demand vs Electricity Price")
plt.xlabel("Price (£/kWh)")
plt.ylabel("Demand (kW)")
plt.grid(True, alpha=0.3)
save_and_show("graph_04_demand_vs_price.png")

# =demand by household
household_mean_demand = df.groupby("household_id")["demand"].mean().sort_values()

plt.figure(figsize=(12, 5))
plt.bar(household_mean_demand.index, household_mean_demand.values)
plt.title("Mean Demand by Household")
plt.xlabel("Household ID")
plt.ylabel("Mean Demand (kW)")
plt.xticks(rotation=90)
plt.grid(True, axis="y", alpha=0.3)
save_and_show("graph_05_mean_demand_by_household.png")

# weekday vs weekend demand
weekday_weekend = df.groupby(["day_category", "hour"])["demand"].mean().unstack(0)

plt.figure(figsize=(10, 5))
for col in weekday_weekend.columns:
    plt.plot(weekday_weekend.index, weekday_weekend[col], marker="o", label=col)
plt.title("Weekday vs Weekend Demand Profile")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
save_and_show("graph_06_weekday_vs_weekend_demand.png")

# demand profile by occupancy type
occ_profile = df.groupby(["occupancy_type", "hour"])["demand"].mean().unstack(0)

plt.figure(figsize=(10, 5))
for col in occ_profile.columns:
    plt.plot(occ_profile.index, occ_profile[col], marker="o", label=col)
plt.title("Demand Profile by Occupancy Type")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
save_and_show("graph_07_demand_by_occupancy_type.png")

# Correlation
numeric_cols = [
    "hour",
    "is_weekend",
    "temperature",
    "price",
    "irradiance",
    "occupancy",
    "demand",
    "solar",
    "base_load",
    "flex_scale",
    "solar_capacity",
    "battery_capacity",
    "hvac_sensitivity",
    "setpoint",
]
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap of Main Variables")
save_and_show("graph_08_correlation_heatmap.png")

# total demand and solar
daily_totals = df.groupby("date")[["demand", "solar"]].sum()

plt.figure(figsize=(12, 5))
plt.plot(daily_totals.index, daily_totals["demand"], label="Daily total demand")
plt.plot(daily_totals.index, daily_totals["solar"], label="Daily total solar")
plt.title("Daily Total Demand and Solar Generation")
plt.xlabel("Date")
plt.ylabel("Total Energy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
save_and_show("graph_09_daily_totals.png")

# household 2 day trace
sample_house = df["household_id"].iloc[0]
house_df = df[df["household_id"] == sample_house].sort_values("timestamp").head(48)

plt.figure(figsize=(12, 6))
plt.plot(house_df["timestamp"], house_df["demand"], label="Demand")
plt.plot(house_df["timestamp"], house_df["solar"], label="Solar")
plt.plot(house_df["timestamp"], house_df["price"], label="Price")
plt.plot(house_df["timestamp"], house_df["temperature"], label="Temperature")
plt.title(f"2-Day Trace for {sample_house}")
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
save_and_show("graph_10_sample_house_2day_trace.png")

# demand boxplot by day type
day_types = sorted(df["day_type"].unique())
box_data = [df[df["day_type"] == d]["demand"] for d in day_types]

plt.figure(figsize=(8, 6))
plt.boxplot(box_data, tick_labels=day_types)
plt.title("Demand Distribution by Day Type")
plt.xlabel("Day Type")
plt.ylabel("Demand (kW)")
plt.grid(True, axis="y", alpha=0.3)
save_and_show("graph_11_demand_boxplot_by_daytype.png")

# mean demand vs mean solar by household
house_summary = df.groupby("household_id").agg(
    mean_demand=("demand", "mean"),
    mean_solar=("solar", "mean"),
    occupancy_type=("occupancy_type", "first")
).reset_index()

plt.figure(figsize=(8, 6))
for occ in house_summary["occupancy_type"].unique():
    temp = house_summary[house_summary["occupancy_type"] == occ]
    plt.scatter(temp["mean_demand"], temp["mean_solar"], label=occ, alpha=0.8)

for _, row in house_summary.iterrows():
    plt.text(row["mean_demand"], row["mean_solar"], row["household_id"], fontsize=8)

plt.title("Household Mean Demand vs Mean Solar")
plt.xlabel("Mean Demand (kW)")
plt.ylabel("Mean Solar (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
save_and_show("graph_12_household_mean_demand_vs_solar.png")

print("Done. All graphs saved as PNG files.")