# Federated Learning to Assist Home Energy Management

## Project Overview

This project implements a software-based simulation of a secure Smart Home Energy Management System (SHEMS). The system uses Long Short-Term Memory (LSTM) forecasting, Deep Reinforcement Learning (DRL), and Federated Learning (FL) to optimise household battery charging and discharging decisions.

The main aim is to reduce energy cost and wasted solar energy while preserving household data privacy by keeping raw household data local during training.

## Key Features

- Data preprocessing and exploratory data analysis for household energy data
- Six-hour LSTM forecasting for household demand and grid electricity price
- Federated Learning simulation using Federated Averaging (FedAvg)
- PPO-based Deep Reinforcement Learning battery control agent
- Battery simulation with charge, discharge, and hold actions
- Comparison against baseline and oracle-style control strategies
- Evaluation using prediction error, cost saving, grid import, and wasted energy metrics
- Graphs and CSV outputs for report evidence

## Coursework Context

Module: CIS3414 Secure Complex Systems  
Coursework: Coursework 2  
Project title: Federated Learning to Assist Home Energy Management  

This repository supports the submitted technical report by providing the source code, notebooks, outputs, and instructions required to reproduce the implementation.

## Repository Structure

```text
.
├── README.md
├── EDA.ipynb
├── FederatedLearningPipeline.ipynb
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
│   ├── results/
│   └── models/
├── requirements.txt
└── report/
    └── Secure_Compex_Systems_CW2_Risky_Biscuits.docx
