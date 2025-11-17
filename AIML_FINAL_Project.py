
'''EV Charging Analytics Project
Clean Structure: Preprocessing,
Supervised Learning, ARIMA,
Unsupervised Learning, Business Logic, UI'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import pickle, json, csv
from tkinter import *
from tkinter import messagebox, ttk
from datetime import datetime
import warnings
import re #For parsing station price

# NEW: Suppress warnings from ARIMA for cleaner output
warnings.filterwarnings("ignore")

# -------- Utility for nice output --------
def print_section(title, sep="="):
    print("\n" + sep*8 + f" {title} " + sep*8)


def print_info(msg): print(f"[INFO] {msg}")
def print_warn(msg): print(f"[WARN] {msg}")

# -------- Global variables for GUI --------
city_encoder = LabelEncoder()
user_encoder = LabelEncoder()
station_avg_prices = pd.DataFrame()
# NEW: Storing both historical and forecast data
city_historical_demand = {}
city_forecasts = {}


# -------- 1. PREPROCESSING --------
print_section("STARTING PROJECT LOAD")
csv_path = r"C:\Users\Karishma C\OneDrive\Desktop\EV_sessions.csv"
df = pd.read_csv(csv_path, parse_dates=['session_start_time', 'session_end_time'])
print_section("DATASET LOADED")
print_info(f"Path: {csv_path}, Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print_section("PREPROCESSING")

# Calculate avg station prices before encoding/modifying
try:
    # NEW: Ensure station_id is treated as a string for grouping
    df['station_id'] = df['station_id'].astype(str)
    station_avg_prices = df.groupby(['location_city', 'station_id'])['price_per_kWh_INR'].mean().round(2)
    station_avg_prices = station_avg_prices.reset_index().sort_values(by=['location_city', 'price_per_kWh_INR'])
    print_info("Calculated average station prices for 'Cheapest Stations' feature.")
except Exception as e:
    print_warn(f"Could not calculate station avg prices: {e}")
    station_avg_prices = pd.DataFrame(columns=['location_city', 'station_id', 'price_per_kWh_INR'])


num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df = df.fillna(0)

# Store categorical encoders globally for the GUI
cat_cols = ['station_id', 'vehicle_type'] # Removed city and user
for col in cat_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Handle city and user separately to save encoders
if 'location_city' in df.columns:
    df['location_city'] = city_encoder.fit_transform(df['location_city'].astype(str))
    print_info(f"Encoded 'location_city'. {len(city_encoder.classes_)} cities found.")
if 'user_id' in df.columns:
    df['user_id'] = user_encoder.fit_transform(df['user_id'].astype(str))
    print_info(f"Encoded 'user_id'. {len(user_encoder.classes_)} users found.")

print_info("Filled missing values and encoded categorical columns.")
print_info("Preprocessing done.")


# -------- 2. SUPERVISED LEARNING --------
print_section("SUPERVISED LEARNING")
features = [c for c in df.select_dtypes(include=[np.float64, np.int64]).columns if c not in ['total_cost_INR','Cluster']]
features = [f for f in features if f in df.columns] # Simple check

X = df[features]
y = df['total_cost_INR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=120, random_state=42)
rf_reg.fit(X_train, y_train)
y_rf = rf_reg.predict(X_test)
rf_r2 = r2_score(y_test, y_rf)
rf_mae = mean_absolute_error(y_test, y_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_rf))
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_lin = linreg.predict(X_test)
lin_r2 = r2_score(y_test, y_lin)
lin_mae = mean_absolute_error(y_test, y_lin)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_lin))
print_section("RANDOM FOREST METRICS", sep="-")
print_info(f"R²: {rf_r2:.3f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print_section("LINEAR REGRESSION METRICS", sep="-")
print_info(f"R²: {lin_r2:.3f}, MAE: {lin_mae:.2f}, RMSE: {lin_rmse:.2f}")


if rf_r2 > lin_r2:
    print_info("Random Forest prediction R² is higher than Linear Regression. Good for nonlinear data.")
else:
    print_info("Linear Regression R² is higher. Your data may be linearly separable or features too correlated.")


print_info("Example predictions for single test sample:")
sample_X, sample_y = X_test.iloc[0], y_test.iloc[0]
sample_X_values = sample_X.values.reshape(1, -1) # Reshape for prediction
print(f'   RandomForest: Y_pred={rf_reg.predict(sample_X_values)[0]:.2f}, Y_actual={sample_y:.2f}')
print(f'   LinearReg:    Y_pred={linreg.predict(sample_X_values)[0]:.2f}, Y_actual={sample_y:.2f}')


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_rf, color='green', alpha=0.4, label=f'Random Forest (R2={rf_r2:.2f})')
plt.scatter(y_test, y_lin, color='orange', alpha=0.4, label=f'LinearReg (R2={lin_r2:.2f})')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Cost')
plt.ylabel('Predicted Cost')
plt.title('Actual vs Predicted: RF vs Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('regression_comparison.png')
plt.close()


with open('demand_model.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
with open('model_columns.json', 'w') as f:
    json.dump(list(X.columns), f)
print_info("Random Forest model and columns saved.")

# -------- 3. DEMAND ANALYSIS (HISTORICAL & FORECAST) --------
print_section("DEMAND ANALYSIS (HISTORICAL & FORECAST)")

if 'session_start_time' in df.columns:
    df['session_hour'] = df['session_start_time'].dt.floor('H')
    df['hour_of_day'] = df['session_start_time'].dt.hour
    df['day_of_week'] = df['session_start_time'].dt.dayofweek
else:
    print_warn("session_start_time column missing. Cannot generate demand plots.")

if 'location_city' in df.columns and 'hour_of_day' in df.columns:
    all_cities = df['location_city'].unique()
    print_info(f"Starting Demand Analysis for {len(all_cities)} cities...")
    
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    for city_id in all_cities:
        city_name = city_encoder.inverse_transform([city_id])[0]
        city_df = df[df['location_city'] == city_id]
        
        if city_df.empty:
            print_warn(f"Skipping City {city_name} (ID: {city_id}): No data.")
            continue
        
        # --- 1. Historical Hourly Demand (for GUI & Plot) ---
        hourly_demand = city_df.groupby('hour_of_day').size()
        # Ensure all 24 hours are present
        hourly_demand = hourly_demand.reindex(range(0, 24), fill_value=0)
        
        if not hourly_demand.empty:
            # Save full historical data for GUI "smart" recommendation
            city_historical_demand[city_id] = hourly_demand
        
            plt.figure(figsize=(10, 5))
            hourly_demand.plot(kind='line', marker='o', grid=True)
            plt.title(f'Average Hourly Charging Demand: {city_name}')
            plt.xlabel('Hour of Day (0-23)')
            plt.ylabel('Average Session Count')
            plt.xticks(range(0, 24))
            plt.tight_layout()
            plt.savefig(f'hourly_demand_city_{city_name}.png')
            plt.close()
            print_info(f"Hourly demand plot saved for City {city_name}.")

        # --- 2. Historical Daily Demand (Plot) ---
        daily_demand = city_df.groupby('day_of_week').size()
        
        if not daily_demand.empty:
            daily_demand.index = daily_demand.index.map(day_map)
            daily_demand = daily_demand.reindex(day_map.values(), fill_value=0)
            
            plt.figure(figsize=(10, 5))
            daily_demand.plot(kind='bar', color='skyblue', grid=True)
            plt.title(f'Average Daily Charging Demand: {city_name}')
            plt.xlabel('Day of Week')
            plt.ylabel('Average Session Count')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'daily_demand_city_{city_name}.png')
            plt.close()
            print_info(f"Daily demand plot saved for City {city_name}.")

        # --- 3. ARIMA Forecast (Plot) ---
        ts = city_df.groupby('session_hour').size().asfreq('H', fill_value=0)
        
        if len(ts) < 50:
            print_warn(f"Skipping ARIMA for City {city_name}: Insufficient time series data.")
            continue
            
        try:
            arima_model = ARIMA(ts, order=(2,0,2))
            arima_fit = arima_model.fit()
            
            forecast_steps = 168 # 7 days
            forecast = arima_fit.forecast(steps=forecast_steps)
            city_forecasts[city_id] = forecast # Save forecast
            
            plt.figure(figsize=(12,6))
            plt.plot(ts.index[-168:], ts[-168:], label='Historical Demand (Last 7 Days)')
            plt.plot(forecast.index, forecast, label='ARIMA Forecast (Next 7 Days)', color='orange', linewidth=2)
            plt.title(f'ARIMA Forecast: Next {forecast_steps} Hours for City {city_name}')
            plt.xlabel('Hour')
            plt.ylabel('Session Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'arima_forecast_city_{city_name}.png')
            plt.close()
            print_info(f"ARIMA forecast plot saved for City {city_name}.")
            
        except Exception as e:
            print_warn(f"ARIMA failed for City {city_name} (ID: {city_id}): {e}")

    # Save data for GUI
    with open('city_historical_demand.pkl', 'wb') as f:
        pickle.dump(city_historical_demand, f)
    print_info("All city historical *demand data* (for GUI) saved.")
    
    # Save ARIMA forecast data
    with open('city_forecasts.pkl', 'wb') as f:
        pickle.dump(city_forecasts, f)
    print_info("All city *ARIMA forecasts* saved.")

else:
    print_warn("Skipping Demand Analysis: 'location_city' or 'session_start_time' column missing.")


# -------- 4. UNSUPERVISED LEARNING (KMeans) --------
print_section("UNSUPERVISED LEARNING (KMeans)")
# Re-check columns
kmeans_data_cols = [c for c in num_cols if c not in ['total_cost_INR', 'computed_duration_min', 'expected_total_cost_INR', 'duration_hours']]
kmeans_data_cols = [f for f in kmeans_data_cols if f in df.columns]
data = df[kmeans_data_cols]

# NEW FEATURE 2: Run KMeans only if data is valid
if not data.empty and len(kmeans_data_cols) > 0:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    wcss = []
    K = range(1, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_data)
        wcss.append(km.inertia_)
    plt.figure(figsize=(7,5))
    plt.plot(K, wcss, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('elbow_method.png')
    plt.close()
    print_info("KMeans Elbow Method plot saved.")
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    silhouette = silhouette_score(scaled_data, df['Cluster'])
    print_info(f"KMeans: k={optimal_k}, Silhouette Score: {silhouette:.4f}")
    cluster_summary = df.groupby('Cluster')[data.columns].mean().round(2)
    print_section("Cluster Summary per Cluster", sep="~")
    print(cluster_summary)
    
    
    print_info("Sample unsupervised cluster assignment:")
    sample_cluster = kmeans.predict([scaled_data[0]])[0]
    print(f"   X={data.iloc[0].values}, Cluster={sample_cluster}")
    
    # Check if columns exist before plotting
    if 'session_duration_min' in df.columns and 'battery_capacity_kWh' in df.columns:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='session_duration_min', y='battery_capacity_kWh', hue='Cluster', data=df, palette='viridis', s=70)
        plt.title('Clusters: Session Duration vs Battery Capacity')
        plt.xlabel('Session Duration (minutes)')
        plt.ylabel('Battery Capacity (kWh)')
        plt.legend(title='Cluster')
        plt.savefig('cluster_duration_battery.png')
        plt.close()
        print_info("KMeans cluster (Duration vs Battery) plot saved.")
    else:
        print_warn("Skipping cluster_duration_battery.png plot: Columns missing.")

    if 'energy_consumed_kWh' in df.columns and 'total_cost_INR' in df.columns:
        plt.figure(figsize=(7,5))
        sns.scatterplot(x='energy_consumed_kWh', y='total_cost_INR', hue='Cluster', data=df, palette='plasma', s=70)
        plt.title('Clusters: Energy Consumed vs Total Cost')
        plt.xlabel('Energy Consumed (kWh)')
        plt.ylabel('Total Cost (INR)')
        plt.legend(title='Cluster')
        plt.savefig('cluster_energy_cost.png')
        plt.close()
        print_info("KMeans cluster (Energy vs Cost) plot saved.")
    else:
        print_warn("Skipping cluster_energy_cost.png plot: Columns missing.")
else:
    print_warn("Skipping KMeans: No numeric data found for clustering.")


# -------- 5. FINAL ALGO / BUSINESS LOGIC --------
USER_DB = {}
BADGE_THRESHOLD = 350
PETROL_AVG_CO2 = 450
PETROL_COST_PER_KWH = 17


def calculate_savings_and_trees(ev_kwh, ev_co2_intensity):
    co2_petrol = ev_kwh * PETROL_AVG_CO2
    co2_ev = ev_kwh * ev_co2_intensity
    co2_saved = co2_petrol - co2_ev
    trees_equiv = co2_saved / 21000 if co2_saved > 0 else 0
    return round(co2_saved, 2), round(trees_equiv, 3)


def petrol_vs_ev_comparison(ev_kwh, ev_cost, ev_co2, trees_equiv):
    petrol_total_cost = ev_kwh * PETROL_COST_PER_KWH
    petrol_total_co2 = ev_kwh * PETROL_AVG_CO2
    ev_total_co2 = ev_kwh * ev_co2
    co2_saved = petrol_total_co2 - ev_total_co2
    better = "EV saves more CO2! 🚗🌱" if co2_saved >= 0 else "Petrol is cleaner for this session! ⛽⚠️"
    result = (f"EV Cost: ₹{ev_cost:.2f} | Petrol Cost: ₹{petrol_total_cost:.2f}\n"
              f"EV CO2: {ev_total_co2:.2f} g | Petrol CO2: {petrol_total_co2:.2f} g\n"
              f"CO2 Saved: {max(0, co2_saved):.2f} g\n"
              f"Trees Equivalent: {max(0, trees_equiv):.3f}\n"
              f"Result: {better}")
    return result


def get_dynamic_price(user_id, base_price, hour, day, city_encoded, renewable_share, carbon_intensity, green_points_to_redeem, model, model_columns, energy_kwh=15):
    processed_input = pd.DataFrame(columns=model_columns)
    processed_input.loc[0] = 0
    
    for col in model_columns:
        if col in ['hour_of_day', 'day_of_week', 'is_weekend', 'location_city']:
            continue
        if col not in processed_input.columns:
            processed_input[col] = 0
            
    processed_input['hour_of_day'] = hour
    processed_input['day_of_week'] = day
    processed_input['is_weekend'] = 1 if day in [5, 6] else 0
    
    if 'location_city' in processed_input.columns:
         processed_input['location_city'] = city_encoded
    
    processed_input = processed_input[model_columns]

    predicted_demand = model.predict(processed_input)[0]
    
    multiplier = 1.0
    if predicted_demand <= 1.3: multiplier = 0.8
    elif predicted_demand >= 1.6: multiplier = 1.5
    
    if carbon_intensity < 350: multiplier *= 0.92
    if renewable_share > 60: multiplier *= 0.95
    
    discount_from_points = min(green_points_to_redeem * 2, base_price * 0.5)
    final_price = max(base_price * multiplier - discount_from_points, 5.0)
    
    points_earned = 2 if carbon_intensity < 350 or renewable_share > 60 else (1 if carbon_intensity < 500 else 0)
    
    if user_id not in USER_DB:
        USER_DB[user_id] = {'green_points': 0, 'sessions': 0, 'avg_co2': 0, 'total_trees': 0}
        
    record = USER_DB[user_id]
    running_total = (record['avg_co2'] * record['sessions'] + carbon_intensity)
    record['sessions'] += 1
    record['green_points'] = max(0, record['green_points'] - green_points_to_redeem + points_earned)
    record['avg_co2'] = running_total / record['sessions']
    
    co2_saved, trees_equiv = calculate_savings_and_trees(energy_kwh, carbon_intensity)
    record['total_trees'] += trees_equiv
    USER_DB[user_id] = record
    
    badge = "🏆 SUPER GREEN" if record['avg_co2'] < BADGE_THRESHOLD else ""
    
    return round(final_price, 2), points_earned, badge, record['green_points'], record['avg_co2'], round(record['total_trees'], 3), co2_saved, trees_equiv, carbon_intensity


# -------- 6. UI FEATURE (Tkinter GUI) --------
class EVGUI:
    def __init__(self, master, model, model_columns, city_forecasts, station_avg_prices, city_encoder, user_encoder, df):
        self.master = master
        self.model = model
        self.model_columns = model_columns
        self.city_forecasts = city_forecasts
        self.station_avg_prices = station_avg_prices
        self.city_encoder = city_encoder
        self.user_encoder = user_encoder
        self.df = df
        
        self.userid_str = None
        self.userid_encoded = None
        self.user_city_encoded = None
        self.user_city_name = "N/A"
        
        master.title("⚡EV Green Charge System⚡")
        master.configure(bg="#daeef7")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Arial", 13), padding=6, background="#89e08d")
        self.style.configure("TLabel", font=("Arial", 12), background="#daeef7")
        self.style.configure("Title.TLabel", font=("Arial", 20, "bold"), background="#daeef7", foreground="#2572c4")
        self.style.configure("Info.TLabel", font=("Consolas", 13), background="#daeef7", foreground="#4ea888")
        self.style.configure("Reco.TFrame", background="#fffbcc", bordercolor="#e6e69c", borderwidth=2)
        self.style.configure("Reco.TLabel", font=("Arial", 11, "italic"), background="#fffbcc", foreground="#5c5c00")
        # NEW FEATURE 3: Style for charge screen alert
        self.style.configure("Alert.TLabel", font=("Arial", 11, "bold"), background="#daeef7", foreground="#c93030")

        self.login_screen()


    def login_screen(self):
        self.clear()
        frame = Frame(self.master, bg="#daeef7")
        frame.pack(pady=50)
        
        lbl = ttk.Label(frame, text="🔑  User Login", style="Title.TLabel")
        lbl.grid(row=0, column=0, pady=12, columnspan=2)
        
        ttk.Label(frame, text="User ID:").grid(row=1, column=0, padx=8, sticky='e')
        self.user_entry = Entry(frame, font=("Arial", 13), width=18)
        self.user_entry.grid(row=1, column=1, pady=8, sticky='w')
        
        btn = Button(frame, text="Login", command=self.do_login, font=("Arial", 13), bg="#89e08d", fg="black", width=12)
        btn.grid(row=2, column=0, columnspan=2, pady=12)


    def do_login(self):
        uid_str = self.user_entry.get().strip()
        if not uid_str:
            messagebox.showerror("Login Failed", "Please enter a User ID")
            return
            
        self.userid_str = uid_str
        
        try:
            self.userid_encoded = self.user_encoder.transform([self.userid_str])[0]
            
            user_sessions = self.df[self.df['user_id'] == self.userid_encoded]
            if not user_sessions.empty:
                common_city_encoded = user_sessions['location_city'].mode()[0]
                self.user_city_encoded = common_city_encoded
                self.user_city_name = self.city_encoder.inverse_transform([common_city_encoded])[0]
                
        except ValueError:
            self.userid_encoded = -1
            self.user_city_name = "N/A"
            print_warn(f"User '{uid_str}' not in original dataset. Treating as new user.")

        
        if self.userid_encoded not in USER_DB and self.userid_encoded != -1:
            USER_DB[self.userid_encoded] = {'green_points': 0, 'sessions': 0, 'avg_co2': 0, 'total_trees': 0}
        elif self.userid_encoded == -1 and self.userid_str not in USER_DB:
             USER_DB[self.userid_str] = {'green_points': 0, 'sessions': 0, 'avg_co2': 0, 'total_trees': 0}
             self.userid_encoded = self.userid_str

        self.home_screen()

    # NEW: Reusable function for getting recommendation strings
    def get_recommendation_string(self, city_encoded, city_name, for_home_screen=False):
        if city_encoded is None or city_encoded not in self.city_forecasts:
            if for_home_screen:
                return "💡 Tip: Charge during off peak hours to spend less and earn more points to redeem!"
            else:
                return "No forecast data available for this city."
        
        try:
            forecast = self.city_forecasts[city_encoded]
            if forecast.empty:
                return "No forecast data available for this city."
            
            # Get forecast for *next 24 hours* for simple recommendation
            forecast_24h = forecast.head(24)
            
            peak_demand = forecast_24h.max()
            peak_hour = forecast_24h.idxmax()
            
            off_peak_demand = forecast_24h.min()
            off_peak_hour = forecast_24h.idxmin()
            
            if for_home_screen:
                return (f"👋 Welcome! For your main city ({city_name}):\n"
                        f"Peak demand is forecast around {peak_hour.strftime('%I %p')} ({peak_demand:.1f} sessions).\n"
                        f"Try charging around {off_peak_hour.strftime('%I %p')} for the lowest demand!")
            else:
                # NEW FEATURE 3c: Return alert string for charge screen
                return (f"Alert for {city_name} (next 24h):\n"
                        f"Peak demand: {peak_hour.strftime('%I %p')}. Try to avoid!\n"
                        f"Off-peak: {off_peak_hour.strftime('%I %p')}. Best time to charge!")

        except Exception as e:
            print_warn(f"Error generating recommendation: {e}")
            return "💡 Tip: Charge during off-peak hours to save!"


    def home_screen(self):
        self.clear()
        
        reco_frame = ttk.Frame(self.master, style="Reco.TFrame", padding=10)
        # Use new reusable function
        reco_message = self.get_recommendation_string(self.user_city_encoded, self.user_city_name, for_home_screen=True)
        reco_label = ttk.Label(reco_frame, text=reco_message, style="Reco.TLabel", justify=LEFT)
        reco_label.pack(fill='x')
        reco_frame.pack(fill='x', padx=10, pady=(10, 0))

        frame = Frame(self.master, bg="#daeef7")
        frame.pack(padx=10, pady=20, fill='both', expand=True)

        record = USER_DB[self.userid_encoded]
        badge = "🏆 SUPER GREEN" if record['avg_co2'] < BADGE_THRESHOLD and record['sessions'] > 0 else ""
        info = (
            f"\n🪪 User: {self.userid_str}\n"
            f"🌱 Green Points: {record['green_points']}\n"
            f"🔄 Charging Sessions: {record['sessions']}\n"
            f"🌎 Avg. CO2 Emitted: {record['avg_co2']:.1f} gCO2/kWh\n"
            f"🌳 Total Trees Planted: {record['total_trees']:.3f}\n"
            f"{badge}\n"
        )
        
        ttk.Label(frame, text=info, style="Info.TLabel").grid(row=0, column=0, columnspan=2)
        
        # NEW FEATURE 3: Removed "Find Cheapest Stations" button
        
        ttk.Button(frame, text="🚗 Start Charge", command=self.charge_screen).grid(row=2, column=0, pady=18, sticky="e", padx=5)
        ttk.Button(frame, text="Logout", command=self.login_screen).grid(row=2, column=1, pady=18, sticky="w", padx=5)
        
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)


    # NEW FEATURE 3: Event handler for when a city is selected in charge_screen
    def on_city_select(self, event):
        city_name = self.city_combo.get()
        if not city_name:
            return

        # --- Feature 3c: Show peak/off-peak alert ---
        try:
            city_encoded = self.city_encoder.transform([city_name])[0]
            alert_msg = self.get_recommendation_string(city_encoded, city_name, for_home_screen=False)
            self.charge_reco_label.config(text=alert_msg)
        except Exception as e:
            self.charge_reco_label.config(text=f"Could not get forecast for {city_name}")
            print_warn(f"Error in on_city_select (alert): {e}")

        # --- Feature 3a: Populate station dropdown ---
        try:
            # Note: station_avg_prices uses the *string name* of the city
            city_stations = self.station_avg_prices[self.station_avg_prices['location_city'] == city_name]
            
            if city_stations.empty:
                self.station_combo['values'] = ["No stations found"]
                self.station_combo.current(0)
                # Manually trigger station select to clear price
                self.on_station_select(None)
                return

            # Create list like "STN01 (Avg: ₹14.50)"
            station_list = [
                f"{row['station_id']} (Avg: ₹{row['price_per_kWh_INR']:.2f})"
                for _, row in city_stations.iterrows()
            ]
            
            self.station_combo['values'] = station_list
            self.station_combo.current(0) # Select the first (cheapest) one
            
            # Manually trigger the event handler to update the price
            self.on_station_select(None) 
            
        except Exception as e:
            print_warn(f"Error in on_city_select (stations): {e}")
            self.station_combo['values'] = ["Error loading stations"]
            self.station_combo.current(0)

    # NEW FEATURE 3b: Event handler for when a station is selected
    def on_station_select(self, event):
        selected_string = self.station_combo.get()
        
        # Use regex to find the price
        match = re.search(r"₹(\d+\.?\d*)", selected_string)
        
        self.price_entry.config(state='normal') # Enable entry to update it
        if match:
            price = match.group(1)
            self.price_entry.delete(0, END)
            self.price_entry.insert(0, price)
        else:
            # Clear price if no station is found or string doesn't match
            self.price_entry.delete(0, END)
            self.price_entry.insert(0, "15.0") # Default
        self.price_entry.config(state='readonly') # Make it readonly again

    def charge_screen(self):
        self.clear()
        frame = Frame(self.master, bg="#daeef7")
        # NEW: Reduced top padding to make space
        frame.pack(padx=18,pady=20)
        
        ttk.Label(frame, text="EV Charging Session", style="Title.TLabel").grid(row=0, column=0, columnspan=2, pady=(0,14))
        
        # --- NEW FEATURE 3 & 3c: Redesigned Controls ---
        
        # City Dropdown
        ttk.Label(frame, text="Station City: ").grid(row=1, column=0, sticky='e', padx=5)
        cities = list(self.city_encoder.classes_)
        default_city = self.user_city_name if self.user_city_name != "N/A" else (cities[0] if cities else "N/A")
        
        self.city_combo = ttk.Combobox(frame, values=cities, font=("Arial", 12), width=18, state="readonly")
        self.city_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        self.city_combo.set(default_city)
        # Bind the event
        self.city_combo.bind("<<ComboboxSelected>>", self.on_city_select)

        # Station Dropdown
        ttk.Label(frame, text="Station: ").grid(row=2, column=0, sticky='e', padx=5)
        self.station_combo = ttk.Combobox(frame, font=("Arial", 12), width=18, state="readonly")
        self.station_combo.grid(row=2, column=1, sticky='w', padx=5, pady=2)
        # Bind the event to update price
        self.station_combo.bind("<<ComboboxSelected>>", self.on_station_select)
        
        # Peak/Off-Peak Alert Label
        self.charge_reco_label = ttk.Label(frame, text="", style="Alert.TLabel", justify=LEFT)
        self.charge_reco_label.grid(row=3, column=0, columnspan=2, pady=(5, 10))

        # --- End of New Features ---

        ttk.Label(frame, text="Hour (0-23): ").grid(row=4, column=0, sticky='e', padx=5)
        self.hour_entry = Entry(frame, font=("Arial", 12), width=7)
        self.hour_entry.insert(0, str(datetime.now().hour))
        self.hour_entry.grid(row=4, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Day (0=Mon,..): ").grid(row=5, column=0, sticky='e', padx=5)
        self.day_entry = Entry(frame, font=("Arial", 12), width=7)
        self.day_entry.insert(0, str(datetime.now().weekday()))
        self.day_entry.grid(row=5, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(frame, text="Base Price (INR):").grid(row=6, column=0, sticky='e', padx=5)
        self.price_entry = Entry(frame, font=("Arial", 12), width=9, state='readonly', bg="#f0f0f0") # Readonly by default
        self.price_entry.grid(row=6, column=1, sticky='w', padx=5, pady=2)
        
        # Grid Info
        curr_renew = float(np.random.randint(20, 100))
        curr_co2 = float(np.random.randint(200, 700))
        ttk.Label(frame, text=f"Grid Renewables: {curr_renew}%\nCO₂ Intensity: {curr_co2} gCO2/kWh", font=("Arial", 12), background="#daeef7", foreground="#883c3c").grid(row=7, column=0, columnspan=2, pady=7)
        
        # Redeem Points
        record = USER_DB[self.userid_encoded]
        ttk.Label(frame, text=f"Redeem Points (Max {record['green_points']}): ").grid(row=8, column=0, sticky='e', padx=5)
        self.redeem_entry = Entry(frame, font=("Arial", 12), width=9)
        self.redeem_entry.insert(0, "0")
        self.redeem_entry.grid(row=8, column=1, sticky='w', padx=5, pady=2)
        
        # Buttons
        Button(frame, text="Run Pricing + Charge", font=("Arial", 13), bg="#56dfad", fg="black", command=lambda: self.process_charge(curr_renew, curr_co2)).grid(row=9, column=0, columnspan=2, pady=14)
        Button(frame, text="Back", command=self.home_screen).grid(row=10, column=0, columnspan=2, pady=4)
        
        # Manually trigger city select to populate stations and alert for the default city
        self.on_city_select(None)


    def process_charge(self, curr_renew, curr_co2):
        try:
            # NEW: Read from new entry/combo widgets
            hour = int(self.hour_entry.get())
            day = int(self.day_entry.get())
            base = float(self.price_entry.get())
            points = int(self.redeem_entry.get())
            city_name = self.city_combo.get()
            
            city_encoded = self.city_encoder.transform([city_name])[0]
            
            max_points = USER_DB[self.userid_encoded]['green_points']
            points = min(points, max_points)
            if points < 0: points = 0
            
        except Exception as e:
            messagebox.showerror("Error", f"Fill all fields with correct values.\n{e}")
            return
            
        energy_kwh = 15
        
        final_price, earned, badge, new_pts, avgCO2, total_trees, co2_saved, trees_equiv, carbon_intensity = get_dynamic_price(
            self.userid_encoded, base, hour, day, city_encoded, curr_renew, curr_co2, points,
            self.model, self.model_columns, energy_kwh
        )
        
        petrol_comparison = petrol_vs_ev_comparison(energy_kwh, final_price, carbon_intensity, trees_equiv)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        log_row = [
            self.userid_str, timestamp, city_name, hour, day, base,
            final_price, points, earned, new_pts, avgCO2,
            co2_saved, trees_equiv, badge
        ]
        
        try:
            with open('session_logs.csv', 'a', newline='', encoding='utf-8') as logfile:
                writer = csv.writer(logfile)
                logfile.seek(0, 2)
                if logfile.tell() == 0:
                    writer.writerow([
                        'UserID', 'Timestamp', 'City', 'Hour', 'Day', 'BasePrice',
                        'FinalPrice', 'PointsUsed', 'PointsEarned', 'GreenPoints',
                        'AvgCO2', 'CO2_Saved_g', 'Equivalent_Trees', 'Badge'
                    ])
                writer.writerow(log_row)
        except Exception as e:
            print_warn(f"Could not write to log file: {e}")

        msg = (
            f"Dynamic Price: ₹{final_price}\nPoints Used: {points}\nPoints Earned: {earned}\n"
            f"Current Green Points: {new_pts}\nAvg. CO2: {avgCO2:.1f} gCO2/kWh\n"
            f"{badge if badge else '—'}\n"
            f"Total Trees Planted: {total_trees}\n"
            f"{petrol_comparison}"
        )
        messagebox.showinfo("Charge Complete", msg)
        self.home_screen()


    def clear(self):
        for widget in self.master.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    print_section("PROJECT SUCCESSFUL")
    print_info("Loading models and data for UI...")
    
    try:
        with open('demand_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        
        with open('city_forecasts.pkl', 'rb') as f:
            city_forecasts = pickle.load(f)
        print_info(f"Loaded {len(city_forecasts)} city forecasts.")

        print_info("Launching EV Charging UI...")
        root = Tk()
        # NEW: Increased height for new UI elements
        root.geometry('590x750')
        
        app = EVGUI(root, model, model_columns, city_forecasts, station_avg_prices, city_encoder, user_encoder, df)
        
        root.mainloop()
        print_section("PROJECT EXITING")
        
    except FileNotFoundError as e:
        print_warn(f"FATAL ERROR: Missing file {e.filename}.")
        print_warn("Please run the script to generate .pkl and .json files before running the UI.")
    except Exception as e:
        print_warn(f"An error occurred: {e}")