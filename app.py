from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to determine the optimal ARIMA hyperparameters (p, d, q)
def find_optimal_arima(sales_data, max_p=5, max_d=2, max_q=5):
    def determine_d(data):
        # Check if the data is constant
        if data.nunique() <= 1:
            return 0
        
        for d in range(max_d + 1):
            diff_data = data.diff(d).dropna()
            if diff_data.nunique() <= 1:  # If the differenced data is constant
                return d
            adf_result = adfuller(diff_data)
            if adf_result[1] <= 0.05:
                return d
        return 0  # Default to 0 if no stationarity is found

    d = determine_d(sales_data)

    # Step 2: Grid search for optimal p, q
    best_aic = float("inf")
    best_params = (0, d, 0)
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            model = ARIMA(sales_data, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_params = (p, d, q)
        except:
            continue

    return best_params

# Function to run Exponential Smoothing model
def run_exponential_smoothing(sales_data, forecast_period):
    model = ExponentialSmoothing(sales_data, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(steps=forecast_period)
    return forecast

# Function to run Moving Average model
def run_moving_average(sales_data, forecast_period):
    window_size = 12  # 12 months moving average
    moving_avg = sales_data.rolling(window=window_size).mean()
    forecast = moving_avg[-1]  # last value is used as the forecast for next period
    return [forecast] * forecast_period

# Function to run Last 12 Months model
def run_last_12_months(sales_data, forecast_period):
    last_value = sales_data[-1]  # last available value
    return [last_value] * forecast_period

# Streamlit application
st.title("Sales Forecasting Application")
st.sidebar.header("Model Selection")

# List of models
models = [
    "ARIMA",
    "STL",
    "Exponential Smoothing (EST)",
    "Moving Average",
    "Last 12 Months",
]
selected_model = st.sidebar.selectbox("Select a Forecasting Model", models)

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    try:
        # Try reading with UTF-8 encoding
        data = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        st.warning("File encoding is not UTF-8. Attempting to read with 'latin1'.")
        data = pd.read_csv(uploaded_file, encoding="latin1")
    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
    else:
        st.write("Uploaded Dataset Preview:")
        st.dataframe(data.head())

        # Dynamic column selection
        columns = data.columns.tolist()
        date_column = st.selectbox("Select Date Column", options=columns)
        sales_column = st.selectbox("Select Sales Column", options=columns)
        product_column = st.selectbox("Select Product Column (Optional)", options=columns + ["None"])

        selected_product = None
        if product_column != "None":
            unique_products = data[product_column].unique()
            selected_product = st.selectbox("Select Product to Forecast", options=unique_products)

        # Automatically parse dates without user input
        try:
            if data[date_column].astype(str).str.isdigit().all():
                data[date_column] = pd.to_datetime(data[date_column], format="%Y%m%d", errors="coerce")
            else:
                data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

            if data[date_column].isna().any():
                st.warning("Some rows have invalid dates. These rows will be dropped.")
                data = data.dropna(subset=[date_column])

            data.set_index(date_column, inplace=True)
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            st.stop()

        # Forecasting Period
        forecast_period = st.number_input("Enter Forecast Period (in days/weeks/months)", min_value=1, step=1)
        frequency = st.selectbox("Frequency of Forecast", ["D", "W", "M"], index=0)

        # Initialize p, d, q to default values
        p = 0
        d = 0
        q = 0

        # Button to determine best-fit ARIMA parameters
        if selected_model == "ARIMA":
            if st.button("Best Fit ARIMA Parameters"):
                try:
                    # Filter by selected product
                    if selected_product is not None:
                        data = data[data[product_column] == selected_product]

                    # Resample based on selected frequency
                    if frequency == "D":
                        data_resampled = data.resample("D").sum()
                    elif frequency == "W":
                        data_resampled = data.resample("W").sum()
                    else:  # Monthly
                        data_resampled = data.resample("M").sum()

                    sales_data = data_resampled[sales_column]

                    # Ensure there's enough variation in the data
                    if sales_data.nunique() > 1:
                        optimal_p, optimal_d, optimal_q = find_optimal_arima(sales_data)

                        # Display optimal values in Streamlit sidebar
                        st.sidebar.write("Optimal ARIMA Parameters (based on AIC):")
                        st.sidebar.write(f"p = {optimal_p}, d = {optimal_d}, q = {optimal_q}")

                        # Populate the sidebar with the optimal ARIMA hyperparameters
                        p = st.sidebar.number_input("ARIMA p (Auto-Regressive)", min_value=0, value=optimal_p)
                        d = st.sidebar.number_input("ARIMA d (Differencing)", min_value=0, value=optimal_d)
                        q = st.sidebar.number_input("ARIMA q (Moving Average)", min_value=0, value=optimal_q)
                    else:
                        st.error("The sales data is constant. Please select data with variation to apply ARIMA.")
                        st.stop()

                except Exception as e:
                    st.error(f"Error determining ARIMA parameters: {e}")

            # Button to run the forecast with selected parameters
            if st.button("Run ARIMA Forecast"):
                try:
                    if selected_product is not None:
                        data = data[data[product_column] == selected_product]

                    # Resample based on selected frequency
                    if frequency == "D":
                        data_resampled = data.resample("D").sum()
                    elif frequency == "W":
                        data_resampled = data.resample("W").sum()
                    else:  # Monthly
                        data_resampled = data.resample("M").sum()

                    sales_data = data_resampled[sales_column]

                    # Check for empty data
                    if sales_data.empty:
                        st.error("The selected data is empty. Check your filters or dataset.")
                        st.stop()

                    # Fit ARIMA model
                    model = ARIMA(sales_data, order=(p, d, q)).fit()
                    forecast = model.forecast(steps=forecast_period)

                    # Generate Forecast Dates
                    freq_map = {"D": "D", "W": "W-SUN", "M": "MS"}
                    last_date = sales_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_period,
                        freq=freq_map[frequency]
                    )

                    # Display Forecast
                    st.write(f"Forecast for next {forecast_period} {frequency}:")
                    forecast_df = pd.DataFrame(
                        {
                            "Forecast Date": forecast_dates,
                            "Forecast Value": forecast.values,
                        }
                    )
                    st.dataframe(forecast_df)

                    # Visualization
                    fig, ax = plt.subplots()
                    sales_data.plot(ax=ax, label="Historical Data")
                    forecast.plot(ax=ax, label="Forecast", linestyle="--")
                    ax.set_title("Sales Forecast")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Sales")
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")

        # Exponential Smoothing model
        if selected_model == "Exponential Smoothing (EST)":
            if st.button("Run Exponential Smoothing Forecast"):
                try:
                    # Filter by selected product
                    if selected_product is not None:
                        data = data[data[product_column] == selected_product]

                    # Resample based on selected frequency
                    if frequency == "D":
                        data_resampled = data.resample("D").sum()
                    elif frequency == "W":
                        data_resampled = data.resample("W").sum()
                    else:  # Monthly
                        data_resampled = data.resample("M").sum()

                    sales_data = data_resampled[sales_column]

                    # Forecast with Exponential Smoothing
                    forecast = run_exponential_smoothing(sales_data, forecast_period)

                    # Generate Forecast Dates
                    last_date = sales_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_period,
                        freq=freq_map[frequency]
                    )

                    # Display Forecast
                    st.write(f"Exponential Smoothing Forecast for next {forecast_period} {frequency}:")
                    forecast_df = pd.DataFrame(
                        {
                            "Forecast Date": forecast_dates,
                            "Forecast Value": forecast,
                        }
                    )
                    st.dataframe(forecast_df)

                    # Visualization
                    fig, ax = plt.subplots()
                    sales_data.plot(ax=ax, label="Historical Data")
                    pd.Series(forecast, index=forecast_dates).plot(ax=ax, label="Forecast", linestyle="--")
                    ax.set_title("Exponential Smoothing Forecast")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Sales")
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
