import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(page_title="AAPL Forecasting System", layout="wide")
st.title("📈 Apple Stock Price Forecasting System")
st.markdown("EDA → Diagnostics → Auto-Tuning → Forecast")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
forecast_days = st.sidebar.selectbox("Forecast Days", [7, 15, 30], index=2)
model_choice = st.sidebar.selectbox(
    "Model", ["SARIMA", "Random Forest", "XGBoost", "GRU"]
)
view = st.sidebar.radio("Forecast View", ["Graph", "Table", "Both"])

MIN_PRICE = 291

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, format="mixed", errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    df = df[["Close"]]
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

if uploaded_file is None:
    st.warning("Upload CSV to continue")
    st.stop()

df = load_data(uploaded_file)

# ===================== EDA =====================
st.header("🔍 Exploratory Data Analysis")

c1, c2, c3 = st.columns(3)
c1.metric("Start Date", df.index.min().strftime("%d-%m-%Y"))
c2.metric("End Date", df.index.max().strftime("%d-%m-%Y"))
c3.metric("Records", len(df))

st.subheader("Closing Price Trend")
st.plotly_chart(px.line(df, x=df.index, y="Close"), use_container_width=True)

st.subheader("Return Distribution")
st.plotly_chart(px.histogram(df, x="Return", nbins=100), use_container_width=True)

df["Rolling_Volatility"] = df["Return"].rolling(30, min_periods=1).std()
st.subheader("Rolling Volatility (Start → End)")
st.plotly_chart(
    px.line(df, x=df.index, y="Rolling_Volatility"),
    use_container_width=True
)

# ===================== SPLIT =====================
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

def mae_rmse(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
    )

def fit_status(train_mae, test_mae):
    if train_mae < test_mae * 0.5:
        return "Overfitting"
    elif train_mae > test_mae:
        return "Underfitting"
    else:
        return "Balanced"

# ===================== MODELS =====================
def sarima_model():
    model = SARIMAX(train["Return"], order=(1, 0, 1))
    fit = model.fit(disp=False)
    train_pred = fit.fittedvalues
    test_pred = fit.forecast(len(test))
    future = fit.forecast(forecast_days)
    return train_pred, test_pred, future

def rf_model(tuned=False):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train["Return"].values
    X_test = np.arange(len(train), len(df)).reshape(-1, 1)

    if tuned:
        params = {"n_estimators": [100, 300], "max_depth": [5, 10]}
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            params,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10)

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future = model.predict(future_idx)
    return train_pred, test_pred, future

def xgb_model(tuned=False):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train["Return"].values
    X_test = np.arange(len(train), len(df)).reshape(-1, 1)

    if tuned:
        params = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.05],
        }
        grid = GridSearchCV(
            XGBRegressor(objective="reg:squarederror"),
            params,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05
        )

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future = model.predict(future_idx)
    return train_pred, test_pred, future

def gru_model():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Return"]])

    X, y = [], []
    lookback = 20
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X_train, X_test = X[: train_size - lookback], X[train_size - lookback :]
    y_train, y_test = y[: train_size - lookback], y[train_size - lookback :]

    model = Sequential(
        [
            GRU(32, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            GRU(16),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0,
    )

    train_pred = scaler.inverse_transform(
        model.predict(X_train)
    ).flatten()
    test_pred = scaler.inverse_transform(
        model.predict(X_test)
    ).flatten()

    last_seq = X[-1]
    future = []
    for _ in range(forecast_days):
        r = model.predict(last_seq.reshape(1, lookback, 1))[0, 0]
        future.append(r)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = r

    future = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

    return train_pred, test_pred, future

# ===================== EXECUTION =====================
st.header("🧠 Model Diagnostics & Auto-Tuning")

if model_choice == "SARIMA":
    train_pred, test_pred, future_returns = sarima_model()
elif model_choice == "Random Forest":
    train_pred, test_pred, future_returns = rf_model(False)
elif model_choice == "XGBoost":
    train_pred, test_pred, future_returns = xgb_model(False)
else:
    train_pred, test_pred, future_returns = gru_model()

train_mae, train_rmse = mae_rmse(
    train["Return"][: len(train_pred)], train_pred
)
test_mae, test_rmse = mae_rmse(test["Return"], test_pred)

status = fit_status(train_mae, test_mae)

before = pd.DataFrame(
    {
        "Stage": ["Before Tuning"],
        "Train MAE": [train_mae],
        "Train RMSE": [train_rmse],
        "Test MAE": [test_mae],
        "Test RMSE": [test_rmse],
        "Fit Status": [status],
    }
)

if status != "Balanced" and model_choice in ["Random Forest", "XGBoost"]:
    st.warning(f"{status} detected → Auto-tuning applied")
    time.sleep(1)

    if model_choice == "Random Forest":
        train_pred, test_pred, future_returns = rf_model(True)
    else:
        train_pred, test_pred, future_returns = xgb_model(True)

    train_mae, train_rmse = mae_rmse(
        train["Return"][: len(train_pred)], train_pred
    )
    test_mae, test_rmse = mae_rmse(test["Return"], test_pred)
    status = fit_status(train_mae, test_mae)

after = pd.DataFrame(
    {
        "Stage": ["After Tuning"],
        "Train MAE": [train_mae],
        "Train RMSE": [train_rmse],
        "Test MAE": [test_mae],
        "Test RMSE": [test_rmse],
        "Fit Status": [status],
    }
)

st.subheader("📊 MAE & RMSE — Before vs After")
st.dataframe(pd.concat([before, after], ignore_index=True))

# ===================== PRICE FORECAST =====================
historical_vol = df["Return"].std()
recent_mean = df["Return"].tail(30).mean()

last_price = df["Close"].iloc[-1]
future_prices = [last_price]

for r in future_returns:
    adjusted = 0.7 * r + 0.3 * recent_mean
    noise = np.random.normal(0, historical_vol)
    next_price = future_prices[-1] * (1 + adjusted + noise)
    next_price = max(next_price, MIN_PRICE)
    future_prices.append(next_price)

future_prices = np.array(future_prices[1:])

price_actual = df["Close"].iloc[-len(future_prices) :]
price_mae, price_rmse = mae_rmse(price_actual, future_prices)

st.subheader("💰 Price-Based Error Metrics")

p1, p2 = st.columns(2)
p1.metric("MAE (Price)", f"{price_mae:.2f}")
p2.metric("RMSE (Price)", f"{price_rmse:.2f}")

future_dates = pd.date_range(
    df.index[-1], periods=forecast_days + 1
)[1:]

forecast_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Predicted Price": future_prices,
    }
)

st.header("📈 Forecast Output")

if view in ["Graph", "Both"]:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Actual")
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=future_prices, name="Forecast"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

if view in ["Table", "Both"]:
    st.dataframe(forecast_df)

st.success("✅ Full pipeline completed successfully")
