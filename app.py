import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import os
import altair as alt


# =========================================================================================================================================
# DATA ENGINEERING
# =========================================================================================================================================

import pandas as pd
import plotly.express as px

def load_data(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Total Volume"] = df["Weight"] * df["Reps"] * df["Sets"]

    df = df.sort_values("Date")
    df["Session ID"] = df["Date"].factorize()[0] + 1
    df = df.set_index("Date")

    return df


# Solid Analytics

#Weekly workload
def calculate_weekly_volume(gym_data):
    weekly = (
        gym_data
        .resample('W-SUN')['Total Volume']
        .sum()
        .reset_index()
    )

    weekly['Rolling Average'] = (
        weekly['Total Volume']
        .rolling(2)
        .mean()
    )

    weekly['Growth Rate'] = (
        weekly['Total Volume']
        .pct_change()
        .fillna(0) * 100
    )

    return weekly


# def vizualizations():

#     #1st Vizualization
#     plot_title = 'Workload Fluctuation Chart'

#     workload_data = px.line(
#         weekly_volume,
#         x = 'Date',
#         y = ['Total Volume', 'Rolling Average'],
#         markers = True,
#         color_discrete_sequence=['#00ADB5','#222831']
#     )

#     workload_data.update_layout(
#         xaxis_title="Week Number",
#         yaxis_title="Total Workload per Week",
#         title = plot_title,
#         font_color = '#393E46',
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         hovermode = 'x unified'


#     #2nd Vizualization
#     weekly_growth_rate = px.line(
#         weekly_volume,
#         x='Date',
#         y= 'Growth Rate',
#         markers=True,
#         title='Weekly Growth Rate',
#         color_discrete_sequence=['#222831']
#     )

#     weekly_growth_rate.update_layout(
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         hovermode = 'x unified',
#         font_color = '#393E46'
#     )

#     weekly_growth_rate.show()

#     #3rd Vizualization
#     weight_growth_rate = px.bar(
#         x = weekly_strength['Date'],
#         y = weekly_strength['Weight'],
#         labels = {'x': 'Date', 'y': 'Average Weight'},
#         color_discrete_sequence = ['#00ADB5']
#     )

#     weight_growth_rate.update_layout(
#         title = 'Average Weight Increase per Week',
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         font_color = '#393E46',
#     )

#     weight_growth_rate.show()

#     #4th Vizualization
#     weight_growth_rate = px.line(
#         weekly_strength,
#         x = weekly_strength['Date'],
#         y = weekly_strength['Growth Rate'],
#         color_discrete_sequence = ['#00ADB5'],
#         markers = True
#     )

#     weight_growth_rate.update_layout(
#         title = 'Strength Growth Rate per Week',
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         font_color = '#393E46',
#     )

#     #5th Vizualization
#     plot_title = 'Training State Map'

#     training_index = px.scatter(
#         training_load,
#         x = 'Volume Z',
#         y = 'Weight Z',
#         color = 'PhaseTag',
#         size = 'Size Metric'
#     )

#     training_index.add_hline(y=0)
#     training_index.add_vline(x=0)

#     training_index.update_layout(
#         title = plot_title,
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         font_color = '#393E46',
#     )

#     #6th Vizualization
#     plot_title = 'Load vs Fatigue Control'

#     fatigue_ratio = px.scatter(
#         fatigue_data,
#         x = 'Chronic Load',
#         y = 'ACWR',
#         color = 'Interpretation'
#     )

#     fatigue_ratio.add_hline(
#         y=1.5, 
#         line_width=2, 
#         line_dash="dash", 
#         line_color="red", 
#         annotation_text="Risk Threshold",

#     )

#     fatigue_ratio.add_hline(
#         y=0.8, 
#         line_width=2, 
#         line_dash="dash", 
#         line_color="#00ADB5"
#     )

#     fatigue_ratio.add_hline(
#         y=1.3, 
#         line_width=2, 
#         line_dash="dash", 
#         line_color="#00ADB5", 
#         annotation_text="Productive Zone"
#     )

#     fatigue_ratio.add_hrect(
#         y0=0.8, y1=1.3,
#         fillcolor="#00ADB5",
#         opacity=0.2,
#         line_width=0
#     )

#     fatigue_ratio.update_layout(
#         title = plot_title,
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         font_color = '#393E46'
#     )

#     #7th Vizualization
#     rm_vs_fat_plot = px.scatter(
#         squat_data,
#         x = '1RM',
#         y = 'Fatigue Ratio'
#     )

#     rm_vs_fat_plot.update_layout(
#         title = plot_title,
#         plot_bgcolor = '#EEEEEE',
#         paper_bgcolor = '#EEEEEE',
#         font_color = '#393E46'
#     )

#     rm_vs_fat_plot.show()
# )


#Average weight per week
def calculate_weekly_avg_weight(gym_data):
    weekly_avg_weight = gym_data.resample('W-SUN')['Weight'].mean().reset_index()
    return weekly_avg_weight


#Weekly growth rate strength
def calculate_weekly_weight_growth_rate(weekly_avg_weight):
    df = weekly_avg_weight.copy()

    df['Growth Rate'] = (
        df['Weight']
        .pct_change()
        .fillna(0) * 100
    )

    return df


def build_training_state(weekly_volume_df, weekly_avg_weight_df):
    df = pd.merge(
        weekly_volume_df,
        weekly_avg_weight_df,
        on='Date',
        how='inner'
    )

    # Z scores
    df['Volume Z'] = (
        df['Total Volume'] - df['Total Volume'].mean()
    ) / df['Total Volume'].std()

    df['Weight Z'] = (
        df['Weight'] - df['Weight'].mean()
    ) / df['Weight'].std()

    df['Training Index'] = df['Volume Z'] + df['Weight Z']
    df['Week_Label'] = df['Date'].dt.strftime("%b %d")

    # Classification
    def classify(row):
        if (row['Volume Z'] > 0) and (row['Weight Z'] > 0):
            return 'Overload'
        elif (row['Volume Z'] > 0):
            return 'Hypertrophy'
        elif (row['Weight Z'] > 0):
            return 'Strength'
        else:
            return 'Deload'

    df['PhaseTag'] = df.apply(classify, axis=1)

    # Size scaling
    ti = df['Training Index']
    denom = ti.max() - ti.min()
    if denom == 0:
        df['Size Metric'] = 20
    else:
        df['Size Metric'] = ((ti - ti.min()) / denom) * 40 + 10

    training_load = df
    return training_load

# # Fatigue and Load Management Logic
# fatigue_data['Total Volume'].ewm(span=4, adjust=False).mean()

def calculate_acwr(weekly_volume_df):

    df = weekly_volume_df.copy()

    df['Chronic Load'] = (
        df['Total Volume']
        .rolling(4, min_periods=1).mean()
        .mean()
    )

    df['ACWR'] = (
        df['Total Volume'] / df['Chronic Load']
    ).fillna(0)

    def interpret(row):
        if 0.8 <= row['ACWR'] <= 1.3:
            return 'Safe-Stimulus'
        elif row['ACWR'] >= 1.5:
            return 'Spike (Injury Risk)'
        else:
            return 'Under-Stimulus'

    df['Interpretation'] = df.apply(interpret, axis=1)

    return df


#Rest day analysis
def rest_day_analysis(gym_data):
    feature_data = pd.DataFrame()
    feature_data['Date'] = gym_data.index
    feature_data['Rest Days'] = gym_data.index.diff()
    feature_data["Rest Days"] = feature_data["Rest Days"].dt.days.fillna(0)
    feature_data['Weekday'] = feature_data['Date'].dt.day_name()
    return feature_data


def rest_day_generation(gym_data, feature_data):

    feature_data = feature_data.reset_index(drop=True)

    gym_data = gym_data.reset_index()

    gym_data = gym_data.merge(
        feature_data[['Date', 'Rest Days']],
        on='Date',
        how='left'
    )

    gym_data = gym_data.set_index('Date')

    return gym_data


def roll_7_day_volume(gym_data):
    daily_volume = gym_data.groupby('Date')['Total Volume'].sum().reset_index()
    daily_volume['Rolling 7 Day Volume'] = (
        daily_volume['Total Volume']
        .rolling(7)
        .sum()
    )

    gym_data = gym_data.merge(
        daily_volume[["Date", "Rolling 7 Day Volume"]],
        on="Date",
        how="left"
    )

    return gym_data, daily_volume


def roll_28_day_volume(daily_volume, gym_data):
    daily_volume['Rolling 28 Day Volume'] = (
        daily_volume['Total Volume']
        .rolling(28)
        .sum()
    )

    gym_data = gym_data.merge(
        daily_volume[["Date", "Rolling 28 Day Volume"]],
        on="Date",
        how="left"
    )
    return gym_data

def add_fatigue_ratio(gym_data):
    gym_data['Fatigue Ratio'] = gym_data['Rolling 7 Day Volume'] / gym_data['Rolling 28 Day Volume']
    return gym_data



#Extract Squat rows and compute 1RM
def extract_squat_1rm(gym_data):
    squat_data = gym_data[gym_data['Exercise'] == 'Squat'].copy()
    squat_data['1RM'] = squat_data['Weight'] * ( 1 + (squat_data['Reps']/30))
    return squat_data



# =========================================================================================================================================
# MODELING
# =========================================================================================================================================

def calculate_1rm(df):
    df = df.copy()
    df['1RM'] = df['Weight'] * (1 + df['Reps'] / 30)
    return df


def prepare_squat_timeseries(df):
    squat_df = df[df['Exercise'] == 'Squat'].copy()
    squat_df = squat_df.sort_values('Date').reset_index(drop=True)
    squat_df['t'] = range(len(squat_df))
    return squat_df


def fit_linear_trend(squat_df):
    df = squat_df.copy()
    squat_df['t'] = range(len(squat_df))
    y = squat_df['1RM']
    X = squat_df['t']
    model = sm.OLS(y, X).fit()

    df['Predicted'] = model.predict(X)
    df['Residual'] = df['1RM'] - df['Predicted']

    return df, model


def get_residual_summary(squat_df):
    residual_summary = squat_df[['t', '1RM', 'Predicted', 'Residual']]
    return residual_summary


# plot_title = 'Residuals vs Time'

# residual_plot = px.scatter(
#     residual_df,
#     x = 't',
#     y = 'Residual',
# )

# residual_plot.add_hline(
#     y=0, 
#     line_width=2, 
#     line_color="#00ADB5"
# )

# residual_plot.update_layout(
#     title = plot_title,
#     plot_bgcolor = '#EEEEEE',
#     paper_bgcolor = '#EEEEEE',
#     font_color = '#393E46',
# )


# residual_plot.show()


def fit_quadratic_trend(squat_df):
    df = squat_df.copy()
    df['t_centered'] = df['t'] - df['t'].mean()
    df['t2'] = df['t_centered']**2
    X = sm.add_constant(df[['t_centered', 't2']])
    model_quad = sm.OLS(df['1RM'], X).fit()
    return df, model_quad


# plot_title = 'Prdicted vs Actual Values'

# multiquad_plot = px.scatter(
#     squat_df,
#     x = 'Predicted',
#     y = '1RM',
# )

# multiquad_plot.add_hline(
#     y=0, 
#     line_width=2, 
#     line_color="#00ADB5"
# )

# multiquad_plot.update_layout(
#     title = plot_title,
#     plot_bgcolor = '#EEEEEE',
#     paper_bgcolor = '#EEEEEE',
#     font_color = '#393E46',
# )

# multiquad_plot.show()


# plot_title = 'Residuals vs Time (Centered)'

# multiquad_plot = px.scatter(
#     squat_df,
#     x = 't_centered',
#     y = 'Residual',
# )

# multiquad_plot.add_hline(
#     y=0, 
#     line_width=2, 
#     line_color="#00ADB5"
# )

# multiquad_plot.update_layout(
#     title = plot_title,
#     plot_bgcolor = '#EEEEEE',
#     paper_bgcolor = '#EEEEEE',
#     font_color = '#393E46',
# )


# multiquad_plot.show()


from sklearn.linear_model import LinearRegression

def train_strength_model(df):
    df = df.fillna(0)

    features = ['Session ID']

    if len(df) < 5:
        raise ValueError("Dataset too small")

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    model = LinearRegression()
    model.fit(train[features], train['1RM'])

    predictions = model.predict(test[features]) if len(test) > 0 else []

    return model, train, test, predictions



def calculate_mae(test, predictions):
    mae_score = mean_absolute_error(test["1RM"], predictions)
    return mae_score



def train_strength_model_fatigue(model, train, test):
    model = LinearRegression()
    features = ["Session ID", "Fatigue Ratio"]
    model.fit(train[features], train["1RM"])
    predictions2 = model.predict(test[features])
    mae2 = mean_absolute_error(test["1RM"], predictions2)
    return predictions2, mae2
    
    
def train_strength_model_fatigue_and_rolling(model, train, test):
    model = LinearRegression()
    features2 = ['Session ID', 'Fatigue Ratio', 'Rolling 7 Day Volume']
    model.fit(train[features2], train['1RM'])
    predictions3 = model.predict(test[features2])
    mae3 = mean_absolute_error(test["1RM"], predictions3)
    return predictions3, mae3, features2

#Forecasting
def forecasting_variables(squat_data):
    last_session = squat_data['Session ID'].max()
    recent_fatigue = squat_data['Fatigue Ratio'].tail(10).mean()
    avg_volume = squat_data['Rolling 7 Day Volume'].mean()
    return last_session, recent_fatigue, avg_volume


def forecast_future(model, last_session, fatigue, volume, features, n_sessions=30):
    future_sessions = np.arange(last_session + 1, last_session + n_sessions + 1)

    future_df = pd.DataFrame({
        "Session ID": future_sessions,
        "Fatigue Ratio": fatigue,
        "Rolling 7 Day Volume": volume
    })

    future_df["Predicted 1RM"] = model.predict(future_df[features])

    return future_df

#Confidence Intervals

def add_confidence_intervals(future_df, train, model, features):

    train_preds = model.predict(train[features])
    residual = train['1RM'] - train_preds
    residual_std = np.std(train['1RM'] - train_preds)

    future_df["Lower CI"] = future_df["Predicted 1RM"] - 1.96 * residual_std
    future_df["Upper CI"] = future_df["Predicted 1RM"] + 1.96 * residual_std

    return future_df, residual_std, residual


def predict_target_session(future_df, target):
    target_df = future_df[future_df["Lower CI"] >= target]

    if len(target_df) > 0:
        return int(target_df['Session ID'].iloc[0])
    return None


def best_weekday(squat_df):
    df = squat_df.copy()

    df['Weekday'] = df['Date'].dt.day_name()

    result = (
        df.groupby('Weekday')['Residual']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return result.iloc[0]['Weekday'], df


def pattern_data(pattern):
    df = pattern
    df['Rest Days'] = df['Date'].diff()
    df['Rest Days'] = df['Rest Days'].dt.days.fillna(0)
    df.groupby('Rest Days')['Session ID'].count()
    return df


def optimal_rest_day(df, residual_std):
    optimal = (
        df.groupby('Rest Days')['Residual']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    if len(optimal) < 3:
        return None

    opt_day = int(optimal['Rest Days'][1])
    second = int(optimal['Rest Days'][2])

    diff = optimal['Residual'][1] - optimal['Residual'][2]
    effect_size = diff / residual_std

    return opt_day, second, effect_size

#Deload Week Recommendation
def deload_recommendation(squat_residual_df, weekly_volume_df):
    rolling_residual = squat_residual_df['Residual'].rolling(3).mean().iloc[-1]
    rolling_fatigue = weekly_volume_df['Fatigue Ratio'].rolling(3).mean().iloc[-1]

    fatigue_mean = weekly_volume_df['Fatigue Ratio'].mean()
    fatigue_std = weekly_volume_df['Fatigue Ratio'].std()

    caution_threshold = fatigue_mean + fatigue_std
    deload_threshold = fatigue_mean + 1.5 * fatigue_std

    if (rolling_fatigue >= deload_threshold) and (rolling_residual < -0.5):
        return "Deload Warning"
    elif (rolling_fatigue >= deload_threshold) or ((rolling_fatigue >= caution_threshold) and (rolling_residual < 0)):
        return "High Fatigue Warning"
    elif (caution_threshold <= rolling_fatigue <= deload_threshold) and (rolling_residual >= 0):
        return "Caution"
    else:
        return "Fatigue Normal"

# =========================================================================================================================================
# STATS
# =========================================================================================================================================

def prepare_stats_data(df):
    data = df.copy()

    data['1RM'] = data['Weight'] * (1 + data['Reps'] / 30)
    data['Session ID'] = data['Date'].factorize()[0] + 1

    return data

def performance_trend(stats_data):
    df = stats_data.copy()

    if len(df) < 40:
        return None

    recent_avg = df['1RM'].iloc[-8:].mean()
    past_avg = df['1RM'].iloc[-38:-30].mean()

    percent_change = ((recent_avg - past_avg) / past_avg) * 100
    percent_change = round(percent_change, 1)

    if percent_change > 5:
        message = "Strong upward trend observed."
    elif 0 < percent_change <= 5:
        message = "Steady progress observed."
    elif -2 < percent_change <= 0:
        message = "Performance stable."
    else:
        message = "Performance decline detected. Recovery review suggested."

    return {
        "recent_avg": round(recent_avg, 2),
        "past_avg": round(past_avg, 2),
        "percent_change": percent_change,
        "message": message
    }


def detect_prs(stats_data):
    df = stats_data.copy()

    session_max = (
        df.groupby(['Exercise', 'Session ID'])['1RM']
        .max()
        .reset_index()
    )

    session_max['Historical Max'] = (
        session_max
        .groupby('Exercise')['1RM']
        .cummax()
        .shift(1)
    )

    session_max['Is PR'] = (
        session_max['1RM'] > session_max['Historical Max']
    )

    session_max['PR%'] = (
        (session_max['1RM'] - session_max['Historical Max'])
        / session_max['Historical Max']
    ) * 100

    session_max['PR%'] = session_max['PR%'].round(1)

    return session_max

def latest_pr_summary(pr_dataframe):
    pr_rows = pr_dataframe[pr_dataframe['Is PR'] == True]

    if pr_rows.empty:
        return None

    latest_pr = pr_rows.groupby('Exercise').tail(1)

    summary = []

    for _, row in latest_pr.iterrows():
        summary.append({
            "Exercise": row['Exercise'],
            "1RM": round(row['1RM'], 1),
            "PR_percent": row['PR%']
        })

    return summary


CSV_PATH = "workouts.csv"

# --------------------------------
# Initialize CSV
# --------------------------------

if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=["Date", "Exercise", "Weight", "Reps", "Sets"])
    df_init.to_csv(CSV_PATH, index=False)

# --------------------------------
# UI
# --------------------------------

st.set_page_config(layout="wide")
st.title("GymIQ")


st.sidebar.header("Data Source")

data_mode = st.sidebar.radio(
    "Choose Data Mode",
    ["User Data", "Synthetic Data"]
)

date = st.sidebar.date_input("Date")
exercise = st.sidebar.text_input("Exercise")
weight = st.sidebar.number_input("Weight (kg)", min_value=0.0)
reps = st.sidebar.number_input("Reps", min_value=1)
sets = st.sidebar.number_input("Sets", min_value=1)

if st.sidebar.button("Add Workout"):
    new_row = pd.DataFrame([{
        "Date": date,
        "Exercise": exercise,
        "Weight": weight,
        "Reps": reps,
        "Sets": sets
    }])
    new_row.to_csv(CSV_PATH, mode="a", header=False, index=False)
    st.sidebar.success("Workout Added")
    st.rerun()

# --------------------------------
# Display Data
# --------------------------------

if data_mode == "User Data":
    raw_df = pd.read_csv(CSV_PATH)
else:
    raw_df = pd.read_csv('C:/Users/Asus/OneDrive/Desktop/GymIQ/data/realistic_synthetic_training_data.csv')

if raw_df.empty:
    st.warning("No workouts added yet.")
else:
    df = load_data(raw_df)

    st.subheader("Workout Log")
    st.dataframe(df.reset_index())


tab1, tab2, tab3, tab4, tab5 = st.tabs(['Overview', 'Volume', 'Strength', 'Fatigue', 'Forecast'])

# --------------------------------
# OVERVIEW
# --------------------------------

with tab1:

    st.subheader('Deload Warning')

    volume = calculate_weekly_volume(df)
    strength = calculate_weekly_avg_weight(df)

    training_state = build_training_state(volume, strength)

    chart_training = (
        alt.Chart(training_state)
        .mark_circle()
        .encode(
            x="Volume Z:Q",
            y="Weight Z:Q",
            size=alt.Size(
                "Size Metric:Q",
                legend= None
            ),
            color="PhaseTag:N",
            tooltip=["PhaseTag"]
        )
        .interactive()
    )

    st.altair_chart(chart_training, use_container_width=True)


    st.subheader("ACWR")

    weekly_vol = calculate_weekly_volume(df)
    acwr = calculate_acwr(weekly_vol)

    chart = (
        alt.Chart(acwr)
        .mark_circle(size=100)
        .encode(
            x="Total Volume:Q",
            y="ACWR:Q",
            color="Interpretation:N",
            tooltip=["Date", "ACWR", "Interpretation"]
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader('Your PRs')

    stats_data = prepare_stats_data(df.reset_index())
    prs = detect_prs(stats_data)
    true_prs = prs[prs['Is PR'] == True]
    latest_prs = (
    true_prs
    .sort_values("Session ID")
    .groupby("Exercise")
    .tail(1)
    )

    for _, row in latest_prs.iterrows():
        st.metric(
            label=row["Exercise"],
            value=f"{round(row['1RM'], 1)} kg",
            delta=f"{row['PR%']}%"
        )

    


    st.subheader('Performance Trend')

    result = performance_trend(stats_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
        "Recent Avg",
        result["recent_avg"],
        delta = result["percent_change"]
    )

    with col2:
        st.metric("Past Avg", result["past_avg"])

    if result['message'] == "Strong upward trend observed.":
        st.success(result["message"])
    elif result['message'] == "Performance decline detected. Recovery review suggested.":
        st.error(result["message"])
    else:
        st.info(result["message"])


# --------------------------------
# VOLUME
# --------------------------------

with tab2:

    st.subheader("Weekly Workload")

    weekly = calculate_weekly_volume(df)

    st.line_chart(
        weekly.set_index("Date")[["Total Volume", "Rolling Average"]]
    )


    st.subheader("Weekly Growth Rate")

    weekly = calculate_weekly_volume(df)

    st.line_chart(
        weekly.set_index("Date")["Growth Rate"]
    )


# --------------------------------
# STRENGTH
# --------------------------------

with tab3:

    st.subheader("Squat 1RM Trend")

    df_1rm = calculate_1rm(df.reset_index())

    squat_df = prepare_squat_timeseries(df_1rm)

    if len(squat_df) < 5:
        st.info("Not enough Squat data for modeling yet.")
    else:
        squat_df, model = fit_linear_trend(squat_df)

        chart_df = squat_df.set_index("t")[["1RM", "Predicted"]]

        st.line_chart(chart_df)

# --------------------------------
# FATIGUE
# --------------------------------

with tab4:

    st.subheader("Fatigue Monitoring")

    weekly = calculate_weekly_volume(df)
    fatigue_data = calculate_acwr(weekly)
    st.line_chart(
        fatigue_data.set_index("Date")[["ACWR"]]
    )


# --------------------------------
# FORECAST
# --------------------------------

with tab5:

    st.subheader('1RM Predictor')
    st.write('This will predict 1RM for your future sessions based on your current performance and outputs the session number in which you are likely to hit your target')
    features = ['Session ID']
   

    df_7, daily = roll_7_day_volume(df)
    df_7_28 = roll_28_day_volume(daily, df_7)
    fat_df = add_fatigue_ratio(df_7_28)
    rm_data = calculate_1rm(fat_df)
    rm_data = rm_data[rm_data["Exercise"] == "Squat"]
    rm_data = rm_data.dropna()
    model, train, test, pred = train_strength_model(rm_data)
    last_session, recent_fatigue, avg_volume = forecasting_variables(rm_data)
    future_df = forecast_future(model, last_session, recent_fatigue, avg_volume, features)
    main_df, residual_std, residual = add_confidence_intervals(future_df, train,model, features)
    target = st.number_input("Enter Target 1RM", min_value=0.0)
    prediction = predict_target_session(main_df, target)

    target_row = main_df[main_df["Session ID"] == prediction]

    st.metric(
        label=f"Predicted Target Session with 95% accuracy",
        value= target_row['Session ID'].values[0]
    )

    st.subheader("Optimal Rest Day Predictor")

    df_1rm = calculate_1rm(df.reset_index())
    squat_df = prepare_squat_timeseries(df_1rm)

    if len(squat_df) < 8:
        st.warning("Not enough squat data for rest-day analysis.")
    else:
        squat_df, model = fit_linear_trend(squat_df)
        pattern_df = pattern_data(squat_df)
        opt_day, second_best_day, effect_size = optimal_rest_day(pattern_df, residual_std)

        st.metric(
            label=f"Optimal Rest Day",
            value =  f"{opt_day} days"
        )
        if effect_size < 0.5:
            st.success(f"Slight performance benefit observed with {opt_day} days recovery compared to {second_best_day} days")
        elif effect_size >= 0.5:
            st.info(f"Better performance observed with {opt_day} days of recovery")
        else:
            st.warning("Need more data to verify the results.")


    squat_1rm_data = extract_squat_1rm(fat_df)
    squat_1rm_data_indexed = squat_1rm_data.set_index(pd.to_datetime(squat_1rm_data['Date']))
    weekly_squat_volume = calculate_weekly_volume(squat_1rm_data_indexed)
    deload_status = deload_recommendation(squat_df, weekly_squat_volume)
    st.write(deload_status)
    
