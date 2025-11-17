# =====================================
#     EV Charging Analytics Project
#     Clean Structure: Preprocessing,
#     Supervised Learning, ARIMA,
#     Unsupervised Learning, Business Logic, UI
# =====================================


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


# -------- Utility for nice output --------
def print_section(title, sep="="):
    print("\n" + sep*8 + f" {title} " + sep*8)


def print_info(msg): print(f"[INFO] {msg}")
def print_warn(msg): print(f"[WARN] {msg}")


# -------- 1. PREPROCESSING --------
print_section("STARTING PROJECT LOAD")
csv_path = input("Enter the path to your dataset CSV file: ")
df = pd.read_csv(csv_path, parse_dates=['session_start_time', 'session_end_time'])
print_section("DATASET LOADED")
print_info(f"Path: {csv_path}, Rows: {df.shape[0]}, Columns: {df.shape[1]}")


print_section("PREPROCESSING")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df = df.fillna(0)
cat_cols = ['station_id', 'location_city', 'vehicle_type', 'user_id']
for col in cat_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
print_info("Filled missing values and encoded categorical columns.")
print_info("Preprocessing done.")


# -------- 2. SUPERVISED LEARNING --------
print_section("SUPERVISED LEARNING")
features = [c for c in df.select_dtypes(include=[np.float64, np.int64]).columns if c not in ['total_cost_INR','Cluster']]
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
print_section("RANDOMFOREST METRICS", sep="-")
print_info(f"R²: {rf_r2:.3f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print_section("LINEARREGRESSION METRICS", sep="-")
print_info(f"R²: {lin_r2:.3f}, MAE: {lin_mae:.2f}, RMSE: {lin_rmse:.2f}")


if rf_r2 > lin_r2:
    print_info("RandomForest prediction R² is higher than LinearRegression. Good for nonlinear data.")
else:
    print_info("LinearRegression R² is higher. Your data may be linearly separable or features too correlated.")


print_info("Example predictions for single test sample:")
sample_X, sample_y = X_test.iloc[0], y_test.iloc[0]
print(f'   RandomForest: X={sample_X.values}, Y_pred={rf_reg.predict([sample_X.values])[0]:.2f}, Y_actual={sample_y:.2f}')
print(f'   LinearReg:    X={sample_X.values}, Y_pred={linreg.predict([sample_X.values])[0]:.2f}, Y_actual={sample_y:.2f}')


plt.figure(figsize=(8,5))
plt.scatter(y_test, y_rf, color='green', alpha=0.4, label=f'Random Forest (R2={rf_r2:.2f})')
plt.scatter(y_test, y_lin, color='orange', alpha=0.4, label=f'LinearReg (R2={lin_r2:.2f})')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Cost')
plt.ylabel('Predicted Cost')
plt.title('Actual vs Predicted: RF vs Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('regression_comparison.png') # <-- ADDED LINE
plt.show()


with open('demand_model.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
with open('model_columns.json', 'w') as f:
    json.dump(list(X.columns), f)
print_info("Random Forest model and columns saved.")


# -------- 3. ARIMA ANALYSIS --------
print_section("ARIMA TIME SERIES ANALYSIS")
df['session_hour'] = df['session_start_time'].dt.floor('H')
main_city = df['location_city'].value_counts().idxmax() if 'location_city' in df.columns else None
city_df = df[df['location_city'] == main_city] if main_city else df
if not city_df.empty:
    ts = city_df.groupby('session_hour').size().asfreq('H', fill_value=0)
    arima_model = ARIMA(ts, order=(2,0,2))
    arima_fit = arima_model.fit()
    forecast_steps = 48
    forecast = arima_fit.forecast(steps=forecast_steps)
    plt.figure(figsize=(12,6))
    plt.plot(ts.index, ts, label='Historical Demand')
    plt.plot(pd.date_range(start=ts.index[-1]+pd.Timedelta(hours=1), periods=forecast_steps, freq='H'), forecast, label='ARIMA Forecast', color='orange')
    plt.title(f'ARIMA Forecast: Next {forecast_steps} Hours for City {main_city}')
    plt.xlabel('Hour')
    plt.ylabel('Session Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('arima_forecast.png') # <-- ADDED LINE
    plt.show()
    print_info("ARIMA forecast plotted.")
else:
    print_warn("Skipping ARIMA: City data is empty or city column missing.")


# -------- 4. UNSUPERVISED LEARNING (KMeans) --------
print_section("UNSUPERVISED LEARNING (KMeans)")
kmeans_data_cols = [c for c in num_cols if c not in ['total_cost_INR', 'computed_duration_min', 'expected_total_cost_INR', 'duration_hours']]
data = df[kmeans_data_cols]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
wcss = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    wcss.append(km.inertia_)
plt.figure(figsize=(7,5))
plt.plot(K, wcss, 'bo-', markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('elbow_method.png') # <-- ADDED LINE
plt.show()
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
silhouette = silhouette_score(scaled_data, df['Cluster'])
print_info(f"KMeans: k={optimal_k}, Silhouette Score: {silhouette:.4f}")
cluster_summary = df.groupby('Cluster')[data.columns].mean().round(2)
print_section("Cluster Summary per Cluster", sep="~")
print(cluster_summary)


print_info("Sample unsupervised cluster assignment:")
sample_cluster = kmeans.predict([scaled_data[0]])[0]
print(f"   X={data.iloc[0].values}, Cluster={sample_cluster}")


plt.figure(figsize=(7,5))
sns.scatterplot(x='session_duration_min', y='battery_capacity_kWh', hue='Cluster', data=df, palette='viridis', s=70)
plt.title('Clusters: Session Duration vs Battery Capacity')
plt.xlabel('Session Duration (minutes)')
plt.ylabel('Battery Capacity (kWh)')
plt.legend(title='Cluster')
plt.savefig('cluster_duration_battery.png') # <-- ADDED LINE
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x='energy_consumed_kWh', y='total_cost_INR', hue='Cluster', data=df, palette='plasma', s=70)
plt.title('Clusters: Energy Consumed vs Total Cost')
plt.xlabel('Energy Consumed (kWh)')
plt.ylabel('Total Cost (INR)')
plt.legend(title='Cluster')
plt.savefig('cluster_energy_cost.png') # <-- ADDED LINE
plt.show()


# -------- 5. FINAL ALGO / BUSINESS LOGIC --------
USER_DB = {}
BADGE_THRESHOLD = 350
PETROL_AVG_CO2 = 450   # gCO2/kWh
PETROL_COST_PER_KWH = 17   # INR per kWh


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


def get_dynamic_price(user_id, base_price, hour, day, city, renewable_share, carbon_intensity, green_points_to_redeem, model, model_columns, energy_kwh=15):
    processed_input = pd.DataFrame(columns=model_columns)
    processed_input.loc[0] = 0
    processed_input['hour_of_day'] = hour
    processed_input['day_of_week'] = day
    processed_input['is_weekend'] = 1 if day in [5, 6] else 0
    city_col = f'location_city_{city}'
    if city_col in processed_input.columns:
        processed_input[city_col] = 1
    predicted_demand = model.predict(processed_input)[0]
    multiplier = 0.8 if predicted_demand <= 1.3 else (1.5 if predicted_demand >= 1.6 else 1.0)
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
    def __init__(self, master, model, model_columns):
        self.master = master
        self.model = model
        self.model_columns = model_columns
        self.userid = None
        master.title("⚡EV Green Charge System⚡")
        master.configure(bg="#daeef7")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Arial", 13), padding=6, background="#89e08d")
        self.style.configure("TLabel", font=("Arial", 12), background="#daeef7")
        self.login_screen()


    def login_screen(self):
        self.clear()
        frame = Frame(self.master, bg="#daeef7")
        frame.pack(pady=50)
        lbl = Label(frame, text="🔑  User Login", font=("Arial", 20, "bold"), bg="#daeef7", fg="#2572c4")
        lbl.grid(row=0, column=0, pady=12, columnspan=2)
        Label(frame, text="User ID:", font=("Arial", 13), bg="#daeef7").grid(row=1, column=0, padx=8)
        self.user_entry = Entry(frame, font=("Arial", 13), width=18)
        self.user_entry.grid(row=1, column=1, pady=8)
        btn = Button(frame, text="Login", command=self.do_login, font=("Arial", 13), bg="#89e08d", fg="black", width=12)
        btn.grid(row=2, column=0, columnspan=2, pady=12)


    def do_login(self):
        uid = self.user_entry.get().strip()
        if not uid:
            messagebox.showerror("Login Failed", "Please enter a User ID")
            return
        self.userid = uid
        if uid not in USER_DB:
            USER_DB[uid] = {'green_points': 0, 'sessions': 0, 'avg_co2': 0, 'total_trees': 0}
        self.home_screen()


    def home_screen(self):
        self.clear()
        record = USER_DB[self.userid]
        frame = Frame(self.master, bg="#daeef7")
        frame.pack(padx=10, pady=35)
        badge = "🏆 SUPER GREEN" if record['avg_co2'] < BADGE_THRESHOLD and record['sessions'] > 0 else ""
        info = (
            f"\n🪪 User: {self.userid}\n"
            f"🌱 Green Points: {record['green_points']}\n"
            f"🔄 Charging Sessions: {record['sessions']}\n"
            f"🌎 Avg. CO2 Emitted: {record['avg_co2']:.1f} gCO2/kWh\n"
            f"🌳 Total Trees Planted: {record['total_trees']:.3f}\n"
            f"{badge}\n"
        )
        Label(frame, text=info, font=("Consolas", 13), bg="#daeef7", fg="#4ea888").grid(row=0, column=0, columnspan=2)
        ttk.Button(frame, text="🚗 Start Charge", command=self.charge_screen).grid(row=1, column=0, pady=18, sticky="e")
        ttk.Button(frame, text="Logout", command=self.login_screen).grid(row=1, column=1, pady=18, sticky="w")


    def charge_screen(self):
        self.clear()
        frame = Frame(self.master, bg="#daeef7")
        frame.pack(padx=18,pady=30)
        Label(frame, text="EV Charging Session", font=("Arial", 17, "bold"), bg="#daeef7", fg="#2b44ac").grid(row=0, column=0, columnspan=2, pady=(0,14))
        Label(frame, text="Station City: ", font=("Arial", 12), bg="#daeef7").grid(row=1, column=0, sticky='e')
        self.city = StringVar(value='0')
        cities = [str(city) for city in sorted(df['location_city'].unique())] if 'location_city' in df.columns else ['0']
        ttk.Combobox(frame, textvariable=self.city, values=cities, font=("Arial", 12), width=14).grid(row=1, column=1)
        Label(frame, text="Hour (0-23): ", font=("Arial", 12), bg="#daeef7").grid(row=2, column=0, sticky='e')
        self.hour = Entry(frame, font=("Arial", 12), width=7)
        self.hour.insert(0, "12")
        self.hour.grid(row=2, column=1)
        Label(frame, text="Day (0=Mon,..): ", font=("Arial", 12), bg="#daeef7").grid(row=3, column=0, sticky='e')
        self.day = Entry(frame, font=("Arial", 12), width=7)
        self.day.insert(0, "3")
        self.day.grid(row=3, column=1)
        Label(frame, text="Base Price (INR):", font=("Arial", 12), bg="#daeef7").grid(row=4, column=0, sticky='e')
        self.price = Entry(frame, font=("Arial", 12), width=9)
        self.price.insert(0, "15")
        self.price.grid(row=4, column=1)
        curr_renew = float(np.random.randint(20, 100))
        curr_co2 = float(np.random.randint(200, 700))
        Label(frame, text=f"Grid Renewables: {curr_renew}%\nCO₂ Intensity: {curr_co2} gCO2/kWh", font=("Arial", 12), bg="#daeef7", fg="#883c3c").grid(row=5, column=0, columnspan=2, pady=(7,7))
        record = USER_DB[self.userid]
        Label(frame, text="Redeem Points: ", font=("Arial", 12), bg="#daeef7").grid(row=6, column=0, sticky='e')
        self.redeem = Entry(frame, font=("Arial", 12), width=9)
        self.redeem.insert(0, "0")
        self.redeem.grid(row=6, column=1)
        Button(frame, text="Run Pricing + Charge", font=("Arial", 13), bg="#56dfad", fg="black", command=lambda: self.process_charge(curr_renew, curr_co2, record['green_points'])).grid(row=7, column=0, columnspan=2, pady=14)
        Button(frame, text="Back", command=self.home_screen).grid(row=8, column=0, columnspan=2, pady=4)


    def process_charge(self, curr_renew, curr_co2, max_points):
        try:
            hour = int(self.hour.get())
            day = int(self.day.get())
            base = float(self.price.get())
            points = int(self.redeem.get())
            points = min(points, USER_DB[self.userid]['green_points'])
        except:
            messagebox.showerror("Error", "Fill all fields with correct values.")
            return
        energy_kwh = 15
        final_price, earned, badge, new_pts, avgCO2, total_trees, co2_saved, trees_equiv, carbon_intensity = get_dynamic_price(
            self.userid, base, hour, day, self.city.get(), curr_renew, curr_co2, points,
            self.model, self.model_columns, energy_kwh
        )
        petrol_comparison = petrol_vs_ev_comparison(energy_kwh, final_price, carbon_intensity, trees_equiv)

        # ---- Append log to CSV ----
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        log_row = [
            self.userid, timestamp, self.city.get(), hour, day, base,
            final_price, points, earned, new_pts, avgCO2,
            co2_saved, trees_equiv, badge
        ]
        with open('session_logs.csv', 'a', newline='', encoding='utf-8') as logfile:
            writer = csv.writer(logfile)
            logfile.seek(0, 2)   # Move to end of file
            if logfile.tell() == 0:
                writer.writerow([
                    'UserID', 'Timestamp', 'City', 'Hour', 'Day', 'BasePrice',
                    'FinalPrice', 'PointsUsed', 'PointsEarned', 'GreenPoints',
                    'AvgCO2', 'CO2_Saved_g', 'Equivalent_Trees', 'Badge'
                ])
            writer.writerow(log_row)
        # ---------------------------

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
    print_info("Launching EV Charging UI...")
    with open('demand_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.json', 'r')as f:
        model_columns = json.load(f)
    root = Tk()
    root.geometry('590x520')
    app = EVGUI(root, model, model_columns)
    root.mainloop()
    print_section("PROJECT EXITING")
