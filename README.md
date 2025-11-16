
EV CHARGING ANALYTICS & DYNAMIC PRICING SYSTEM

An end-to-end Machine Learning + Time Series + GUI based system that predicts EV charging cost, performs demand forecasting, identifies user behavior clusters, computes CO₂ savings, applies dynamic pricing, and presents everything with a Tkinter dashboard.

Project Overview
This project provides a complete pipeline for EV charging analytics:
* Preprocessing & Feature Engineering
* Machine Learning (Random Forest & Linear Regression)
* Time Series Analysis (ARIMA)
* Unsupervised Learning (KMeans)
* Dynamic Pricing Engine
* CO₂ & Environmental Impact Calculator
* Reward System with Points & Badges
* Tkinter-based Graphical User Interface
* Automated Session Logging

Dataset Details
1. Main Dataset — EV Charging Sessions

Session Metadata
* session_start_time — Charging session start timestamp
* session_end_time — Charging session end timestamp
* session_duration_min — Duration in minutes (reported)
* station_id — Charging station identifier
* location_city — City of the charging point (Pune/Mumbai etc.)
* vehicle_type — EV category (2W / 3W / 4W)
* battery_capacity_kWh — Vehicle battery size
* charging_power_kW — Charger rated power
* energy_consumed_kWh — Total energy delivered during the session

Environmental & Grid Conditions
* ambient_temperature_C — Temperature (°C)
* humidity_% — Humidity (%)
* renewable_share_% — Share of solar/wind energy in grid
* grid_load_MW — Real-time grid load (MW)
* traffic_index — Traffic congestion index

Pricing & Billing
* price_per_kWh_INR — Base electricity cost per kWh
* total_cost_INR — Final billed cost
* expected_total_cost_INR — ML-predicted cost
* cost_diff — Difference: actual − expected cost
* implied_cost_per_kWh — Derived cost rate
* cost_anomaly_flag — True if cost mismatch is abnormal

Computed / Engineered Features
* event_day — Numerical day indicator
* computed_duration_min — Duration recomputed from timestamps
* duration_diff_min — Difference from reported duration
* duration_hours — Duration in hours
* max_possible_energy_kWh — Max energy based on charger power × time
* energy_exceeds_max_flag — True if reported energy exceeds logical max

User Features
* user_id — User identifier
* hour_of_day — Hour extracted from start time
* day_of_week — 0–6 (Mon–Sun)
* is_weekend — 1 if weekend else 0
* session_date — Date only (YYYY-MM-DD)
* holiday_flag — 1 if a public holiday
* holiday_or_weekend — Combined indicator

Battery SOC (State of Charge)
* start_soc_% — Starting battery percentage
* end_soc_% — Ending battery percentage
* expected_energy_from_soc_kWh — Expected energy from SOC difference
* energy_vs_expected_kWh — Actual − expected energy
* energy_vs_max_pct — % of theoretical maximum consumption
* max_charger_power_kW — Maximum possible charger output

Anomaly Detection
* anomaly_flag — True/False indicator
* anomaly_reasons — Explanation for anomaly
* anomaly_score — Model-generated anomaly probability

Rewards & Gamification
* points_redeemed — Points used this session
* points_earned — Points earned
* trees_saved (if calculated by you) — CO₂-equivalent tree impact
* badge_status (if used) — Green / Super Green badge

Payment Details
* payment_method — UPI / Card / Wallet / Cash
* charging_session_status — success / failed / interrupted

2. CO₂ Mapping Dataset
Used for environmental impact:
* tree_absorption_rate = 21000 gCO₂/year
* petrol_emission_factor = 2392 gCO₂/litre
* Converts EV usage to “trees saved”.
3. Generated Session Logs
session_logs.csv automatically stores:
* User ID
* Timestamp
* Raw + adjusted price
* Points used/earned
* CO₂ saved
* Trees equivalent
* Badge status

Key Features
01. CO₂ & Environmental Impact Calculator
* EV vs Petrol CO₂ comparison
* Calculates:
    * Total EV CO₂
    * Equivalent petrol CO₂
    * CO₂ saved
    * Trees-equivalent impact
* Helps promote eco-friendly charging habits

02. Reward System (Green Points)
* +1 to +2 points per eco-friendly session
* Points redeemable for future discounts
* Tracks:
    * Total sessions
    * Average CO₂ footprint
    * Trees saved
* Awards Super Green Badge

03. Dynamic Pricing Engine
Uses ML + renewable energy + grid load:
* High demand → price increases
* Low demand → discount
* High renewable energy → extra discount
* Low carbon intensity → eco-discount
* Ensures fairness: Minimum price = ₹5

04. Automated Session Logging
Every session saved with:
* Price before/after discount
* Average CO₂
* Reward points
* Badge status
* Timestamp
Useful for analytics, dashboards, audits.

🔧 Technical Workflow
1. Data Acquisition
Load EV dataset from CSV.
2. Preprocessing
* Handle missing values
* Encode categorical variables
* Parse timestamps
* Create hour/day/week features
3. Supervised Learning
* Random Forest Regression
* Linear Regression Predicts: Total Charging Cost (INR)
4. Time Series Forecasting
* ARIMA (2,0,2)
* Predicts hourly charging demand
5. Clustering
* StandardScaler + KMeans (3 clusters)
* Groups users by:
    * Energy usage
    * Duration
    * CO₂ impact
    * Time of day
6. Dynamic Pricing Engine
Combines:
* ML prediction
* Renewable %
* Station load
* Carbon intensity
* Reward points
7. GUI (Tkinter)
* Login screen
* Dashboard
* Charging simulator
* Dynamic price display
* Updated logs

System Flowchart
(As shown in your slide — add your image in README)

Future Scope
* Integration with real-time API (CERC, POSOCO)
* Mobile app version (Flutter/React Native)
* Predictive maintenance for charging stations
* Integration with smart grid pricing
* Blockchain-based carbon credit allocation
* Load balancing between multiple EV chargers

Conclusion (Points)
* The project provides a complete EV analytics framework combining ML, clustering, and time-series forecasting.
* Dynamic pricing ensures fair, optimized, real-time cost adjustment.
* CO₂ calculator and rewards system encourage green behavior.
* ARIMA forecasting helps predict future grid demand.
* Tkinter GUI makes the system easy to use and deploy.
* Overall, this system builds the foundation for smart, sustainable EV charging infrastructure.

References
Research & Official Sources
* Government of India EV Statistics – Ministry of Power
* CEEW India Renewable Energy Dashboard
* POSOCO: National Load Despatch Centre Reports
* IPCC Carbon Emission Factors
* Bureau of Energy Efficiency (BEE) – EV Guidelines
Related Kaggle Datasets
* Electric Vehicle Charging Dataset – Kaggle
* Electric Vehicle Population Data – Kaggle
* EV Charging Behaviour Dataset – Kaggle
* Electric Cars: Energy Consumption Dataset – Kaggle
* Global Power Plant & CO₂ Dataset – Kaggle










