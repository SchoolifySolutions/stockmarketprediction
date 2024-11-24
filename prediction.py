import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

st.title("Stock Price Prediction App 📈")
st.write("This app predicts future stock prices using historical data and Linear Regression.")
st.write("By: Varshith Gude and Sasidhar Jasty")


ticker = st.text_input("Enter the stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Fetch Data"):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found. Check the ticker symbol and date range.")
        else:
            st.success(f"Fetched {ticker} stock data!")
            
            st.write("### Historical Stock Data", data.tail())
            
            st.write("### Stock Price Trend")
            plt.figure(figsize=(10, 5))
            plt.plot(data['Close'], label="Close Price")
            plt.title(f"{ticker} Closing Price")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            data['Date'] = data.index
            data['Date'] = data['Date'].apply(lambda x: x.toordinal())  # Convert date to ordinal
            X = np.array(data['Date']).reshape(-1, 1)  # Dates as features
            y = np.array(data['Close'])  # Closing prices as labels

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            st.write(f"### Model RMSE: {rmse:.2f}")

            st.write("### Actual vs Predicted Prices")
            plt.figure(figsize=(10, 5))
            plt.scatter(X_test, y_test, color='blue', label="Actual Prices")
            plt.scatter(X_test, predictions, color='red', label="Predicted Prices")
            plt.xlabel("Date (ordinal)")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            future_days = st.slider("Predict stock price for the next 'n' days:", min_value=1, max_value=30, value=7)

            if future_days > 0:
                try:
                    if len(data) > 0:
                        last_date = data.index[-1]
                    else:
                        st.error("No data available to predict future prices.")
                        raise ValueError("Empty data index.")

                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
                    future_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)

                    st.write("Future Dates:", future_dates)
                    st.write("Future Ordinals Shape:", future_ordinals.shape)

                    future_predictions = model.predict(future_ordinals)

                    future_predictions = future_predictions.flatten()

                    st.write(f"### Predicted Prices for the next {future_days} days:")
                    future_df = pd.DataFrame({
                        "Date": future_dates, 
                        "Predicted Price (USD)": future_predictions 
                    })
                    st.write(future_df)

                    st.write("### Future Predictions")
                    plt.figure(figsize=(10, 5))
                    plt.plot(future_dates, future_predictions, marker='o', label="Future Predictions")
                    plt.title("Future Stock Price Prediction")
                    plt.xlabel("Date")
                    plt.ylabel("Price (USD)")
                    plt.legend()
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Error predicting future prices: {e}")


    except Exception as e:
        st.error(f"Error fetching data: {e}")
