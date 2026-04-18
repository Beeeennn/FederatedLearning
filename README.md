# Smart Home Energy Dataset

## Overview
This dataset was constructed for smart home energy optimisation using:
- IDEAL Household Energy Dataset (demand + weather)
- Synthetic solar generation (physics-inspired model)
- BMRS Market Index electricity prices

## Data Sources

### 1. Electricity Demand
Derived from IDEAL smart meter data.
Converted from power (W) to energy (kWh) and aggregated hourly.

### 2. Weather Data
Extracted from IDEAL dataset.
Temperature converted from tenths of °C to °C.

### 3. Solar Generation (Synthetic)
Based on PVWatts-style model:
P = P_rated * (G / 1000) * (1 + gamma * (T_cell - 25))

Includes:
- time-of-day irradiance approximation
- seasonal scaling
- weather attenuation
- system losses

Reference:
Dobos, A. P. (2014). PVWatts Version 5 Manual. NREL.

### 4. Electricity Prices
Source: Elexon BMRS Market Index API
- 30-min settlement data
- filtered to APXMIDP provider
- converted £/MWh → £/kWh
- resampled to hourly

## Final Features

Core:
- timestamp
- homeid
- demand_kwh
- solar_kwh
- price_kwh

Temporal:
- hour
- day_of_week
- month
- is_weekend

Optional:
- outdoor_temperature

## Usage

Suitable for:
- LSTM (forecasting)
- DRL (battery optimisation)
- Federated Learning (multi-home training)

## Notes
- Solar is conservative due to UK conditions and averaging
- Price data aligned to dataset timestamps
- Weather conditions simplified
