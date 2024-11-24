import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
from textblob import TextBlob

# Set up the Streamlit app
st.title("Stock Price Prediction and Opinion Analysis App 📈")
st.write("This app predicts future stock prices using historical data and Linear Regression. Additionally, you can perform opinion analysis on stock-related text!")

# Section for Stock Price Prediction (same as before)
ticker = st.text_input("Enter the stock ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Fetch Data"):
    # Fetch historical stock data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found. Check the ticker symbol and date range.")
        else:
            st.success(f"Fetched {ticker} stock data!")
            
            # Display data
            st.write("### Historical Stock Data", data.tail())
            
            # Plot stock prices
            st.write("### Stock Price Trend")
            plt.figure(figsize=(10, 5))
            plt.plot(data['Close'], label="Close Price")
            plt.title(f"{ticker} Closing Price")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            # Prepare data for prediction
            data['Date'] = data.index
            data['Date'] = data['Date'].apply(lambda x: x.toordinal())  # Convert date to ordinal
            X = np.array(data['Date']).reshape(-1, 1)  # Dates as features
            y = np.array(data['Close'])  # Closing prices as labels

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            st.write(f"### Model RMSE: {rmse:.2f}")

            # Plot predictions
            st.write("### Actual vs Predicted Prices")
            plt.figure(figsize=(10, 5))
            plt.scatter(X_test, y_test, color='blue', label="Actual Prices")
            plt.scatter(X_test, predictions, color='red', label="Predicted Prices")
            plt.xlabel("Date (ordinal)")
            plt.ylabel("Price (USD)")
            plt.legend()
            st.pyplot(plt)

            # Predict future prices
            future_days = st.slider("Predict stock price for the next 'n' days:", min_value=1, max_value=30, value=7)

            if future_days > 0:
                try:
                    # Ensure data.index[-1] is valid
                    if len(data) > 0:
                        last_date = data.index[-1]
                    else:
                        st.error("No data available to predict future prices.")
                        raise ValueError("Empty data index.")

                    # Generate future ordinal dates
                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
                    future_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)

                    # Predict prices
                    future_predictions = model.predict(future_ordinals)
                    future_predictions = future_predictions.flatten()

                    # Display future predictions
                    st.write(f"### Predicted Prices for the next {future_days} days:")
                    future_df = pd.DataFrame({
                        "Date": future_dates,  # Ensure future_dates is a list, which is 1-dimensional
                        "Predicted Price (USD)": future_predictions  # Flattened predictions to ensure 1D array
                    })
                    st.write(future_df)

                    # Plot future predictions
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

# Sentiment Analysis Section
st.write("### Opinion Analysis of Stock-Related Text 📊")

# Get user input text (such as stock news, tweets, etc.)
user_input = st.text_area("Enter text to analyze opinion", "Tesla stock looks great for the long term!")

# Perform sentiment analysis when the button is clicked
if st.button("Analyze Opinion"):
    if user_input:
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(user_input)
        sentiment_score = blob.sentiment.polarity  # -1: Negative, 0: Neutral, 1: Positive

        # Determine sentiment category
        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Display the sentiment result
        st.write(f"Opinion: {sentiment}")
        st.write(f"Opinion Score: {sentiment_score:.2f}")

        # Display a simple visualization of the sentiment score
        st.write("### Opinion Score Visualization")
        plt.figure(figsize=(5, 2))
        plt.barh([0], [sentiment_score], color='green' if sentiment_score > 0 else 'red')
        plt.xlim(-1, 1)
        plt.yticks([])
        plt.xlabel("Opinion Score")
        st.pyplot(plt)

    else:
        st.write("Please enter some text to analyze opinion.")
