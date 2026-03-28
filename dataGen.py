import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================
NUM_HOUSES = 20
NUM_DAYS = 60
START_DATE = datetime(2026, 1, 1)

np.random.seed(42)

# ==============================
# HELPERS
# ==============================

def is_weekend(dt):
    return dt.weekday() >= 5

def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2)

def activity_profile(hour, morning_shift, evening_shift):
    morning = gaussian(hour, 7 + morning_shift, 1.5)
    evening = gaussian(hour, 19 + evening_shift, 2.0)
    return 0.2 + 0.8 * morning + 1.2 * evening

def occupancy_rule(occ_type, hour, weekend):
    if occ_type == "working":
        if not weekend and 9 <= hour < 17:
            return 0
        return 1
    elif occ_type == "remote":
        return 1
    elif occ_type == "family":
        if not weekend and 9 <= hour < 15:
            return np.random.choice([0,1], p=[0.7,0.3])
        return 1
    elif occ_type == "retired":
        return 1
    return 1

# ==============================
# STEP 1: TIME INDEX
# ==============================

timestamps = [START_DATE + timedelta(hours=i) for i in range(NUM_DAYS * 24)]
days = sorted(set([t.date() for t in timestamps]))

# ==============================
# STEP 2: DAY TYPES
# ==============================

day_types = {}
for d in days:
    r = np.random.rand()
    if r < 0.25:
        day_types[d] = "sunny"
    elif r < 0.5:
        day_types[d] = "mild"
    elif r < 0.75:
        day_types[d] = "cloudy"
    else:
        day_types[d] = "rainy"

# ==============================
# STEP 3: DAILY WEATHER PARAMS
# ==============================

daily_params = {}

for d in days:
    dtype = day_types[d]

    if dtype == "sunny":
        solar_factor = 1.0
        temp_offset = 3
    elif dtype == "mild":
        solar_factor = 0.7
        temp_offset = 1
    elif dtype == "cloudy":
        solar_factor = 0.4
        temp_offset = 0
    else:
        solar_factor = 0.2
        temp_offset = -2

    base_temp = 8  # winter baseline

    max_temp = base_temp + temp_offset + np.random.normal(0, 1)
    min_temp = max_temp - np.random.uniform(4, 8)

    daily_params[d] = {
        "solar_factor": solar_factor,
        "max_temp": max_temp,
        "min_temp": min_temp
    }

# cobnstant average data
global_data = []

for t in timestamps:
    d = t.date()
    hour = t.hour

    params = daily_params[d]

    # Temperature
    temp_curve = 0.5 * (1 + np.sin((hour - 6) / 24 * 2 * np.pi))
    temp = params["min_temp"] + (params["max_temp"] - params["min_temp"]) * temp_curve
    temp += np.random.normal(0, 0.3)

    # Price
    if hour < 6:
        price = 0.12
    elif hour < 16:
        price = 0.18
    elif hour < 21:
        price = 0.30
    else:
        price = 0.16
    price += np.random.normal(0, 0.003)

    # Solar irradiance
    sunrise, sunset = 8, 16
    if hour < sunrise or hour > sunset:
        irradiance = 0
    else:
        progress = (hour - sunrise) / (sunset - sunrise)
        irradiance = np.sin(np.pi * progress) * params["solar_factor"]

    global_data.append({
        "timestamp": t,
        "hour": hour,
        "day_of_week": t.weekday(),
        "is_weekend": int(is_weekend(t)),
        "day_type": day_types[d],
        "temperature": temp,
        "price": price,
        "irradiance": irradiance
    })

global_df = pd.DataFrame(global_data)

#set constants per house

households = []

occ_types = ["working", "remote", "family", "retired"]
house_types = ["small", "medium", "large"]

for i in range(NUM_HOUSES):
    hid = f"H{i:03d}"

    occ_type = np.random.choice(occ_types)
    house_type = np.random.choice(house_types)

    if house_type == "small":
        base_load = np.random.uniform(0.2, 0.4)
        flex = np.random.uniform(0.5, 1.0)
    elif house_type == "medium":
        base_load = np.random.uniform(0.4, 0.7)
        flex = np.random.uniform(1.0, 1.5)
    else:
        base_load = np.random.uniform(0.7, 1.2)
        flex = np.random.uniform(1.5, 2.5)

    pv = np.random.rand() < 0.6
    battery = np.random.rand() < 0.5

    solar_cap = np.random.uniform(2, 6) if pv else 0
    battery_cap = np.random.uniform(5, 12) if battery else 0

    households.append({
        "household_id": hid,
        "occupancy_type": occ_type,
        "base_load": base_load,
        "flex_scale": flex,
        "solar_capacity": solar_cap,
        "battery_capacity": battery_cap,
        "hvac_sensitivity": np.random.uniform(0.05, 0.15),
        "setpoint": np.random.uniform(19, 22),
        "noise": np.random.uniform(0.02, 0.07),
        "morning_shift": np.random.randint(-1, 2),
        "evening_shift": np.random.randint(-2, 3)
    })

households_df = pd.DataFrame(households)

#Generate the data
rows = []

for _, h in households_df.iterrows():
    soc = h["battery_capacity"] * 0.5

    for _, g in global_df.iterrows():

        occ = occupancy_rule(
            h["occupancy_type"],
            g["hour"],
            g["is_weekend"]
        )

        activity = occ * h["flex_scale"] * activity_profile(
            g["hour"],
            h["morning_shift"],
            h["evening_shift"]
        )

        temp_gap = max(0, abs(g["temperature"] - h["setpoint"]) - 1.5)
        hvac = h["hvac_sensitivity"] * temp_gap

        appliance = 0
        if occ and 18 <= g["hour"] <= 21 and np.random.rand() < 0.5:
            appliance += np.random.uniform(0.5, 1.5)

        demand = max(0, h["base_load"] + activity + hvac + appliance + np.random.normal(0, h["noise"]))

        solar = h["solar_capacity"] * g["irradiance"] * np.random.uniform(0.9, 1.0)

        rows.append({
            "timestamp": g["timestamp"],
            "household_id": h["household_id"],
            "hour": g["hour"],
            "day_of_week": g["day_of_week"],
            "is_weekend": g["is_weekend"],
            "day_type": g["day_type"],
            "temperature": g["temperature"],
            "price": g["price"],
            "irradiance": g["irradiance"],
            "occupancy": occ,
            "demand": demand,
            "solar": solar
        })

timeseries_df = pd.DataFrame(rows)


#save the files
households_df.to_csv("households.csv", index=False)
timeseries_df.to_csv("timeseries.csv", index=False)

print("✅ Generated households.csv and timeseries.csv")