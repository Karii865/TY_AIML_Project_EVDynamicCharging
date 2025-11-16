# TY_AIML_Project_EV_DynamicCharging
EV Charging Analytics & Dynamic Pricing System

*1. Project Title & Objective*
Title: Smart EV Charging Analytics with Dynamic Pricing, CO₂ Impact Calculator & Reward System
Objective:
To develop a machine-learning–powered EV charging system that predicts energy demand, calculates dynamic charging prices, estimates environmental impact (CO₂ savings), and rewards users with Green Points. The system integrates supervised learning, time-series forecasting, clustering, and a Tkinter-based GUI for real-time charging simulation.

*2. Dataset Details*
The project uses a synthetic + enhanced realistic EV charging dataset, containing:
User ID
City & station info
Start/end time
Charging duration
kWh consumed
Base charging cost
Renewable energy %
Carbon intensity
Peak/off-peak indicator
Additional datasets used for modeling & benchmarking:
Hourly electricity demand dataset (for ARIMA forecasting)
EV charging session datasets (Kaggle)
Grid renewable energy & CO₂ factors (CEA/BEE India)
Total records: ~2500+ realistic sessions

*3. Algorithm / Models Used*
Preprocessing
Missing value handling
Timestamp parsing + feature engineering (hour, day, weekday)
Label encoding & StandardScaler normalization
ML Models
Random Forest Regression → Predict total charging cost
Linear Regression → Baseline comparison
ARIMA (2,0,2) → Hourly energy demand forecasting
KMeans (k=3) → User/session behavior clustering
Dynamic Pricing Engine
Combines:
ML prediction
Demand forecast
Renewable energy %
Carbon intensity
Reward points (Green Points)
Ensures minimum price: ₹5

*4. Results (Accuracy, Graphs, etc.)*
Random Forest R² Score: ~0.89
Linear Regression R² Score: ~0.72
ARIMA Forecast: Predicts hourly demand with stable trend recognition
KMeans: Clear segmentation into light, moderate, heavy users
Visual outputs include:
Cost prediction plots
Actual vs Predicted curves
ARIMA forecast graph
Cluster visualization
GUI screenshots

*5. Conclusion*
The system accurately predicts charging cost using ML models.
Dynamic pricing makes charging cost more fair, grid-friendly & eco-adaptive.
CO₂ calculator motivates environmentally responsible EV charging.
Reward system increases user engagement and sustainable behavior.
GUI provides an easy-to-use simulation interface for real-time experimentation.

*6. Future Scope*
Integration with real-time API-based grid data
Mobile app version with live notifications
Solar-powered charging recommendations
Smart routing to cheapest EV station
Load balancing for large-scale EV fleets
Blockchain-based reward tokenization

*7. References*
Kaggle – EV Charging Station Dataset
Kaggle – EV Charging Sessions Data
Kaggle – Electricity Load Forecasting Dataset
Kaggle – Smart Grid Stability Dataset
Central Electricity Authority (CEA), India
Bureau of Energy Efficiency (BEE) – CO₂ Emission Factors
Research papers on EV load prediction & dynamic pricing
Python libraries: pandas, numpy, sklearn, statsmodels, matplotlib, tkinter
