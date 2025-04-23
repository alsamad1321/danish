import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

class ECommerceForecasting:
    def __init__(self):
        # Initialize key components
        self.model = None
        self.scaler = None
    
    def load_sample_data(self):
        """
        Create synthetic sales data for demonstration
        
        Returns:
        pandas.DataFrame: Sample sales data
        """
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        sales = pd.Series(
            [1000 + 50 * i + np.random.normal(0, 100) for i in range(len(dates))],
            index=dates
        )
        df = pd.DataFrame({'date': dates, 'sales': sales})
        return df

    def preprocess_data(self, df):
        """
        Preprocess sales data by creating time-based features
        
        Parameters:
        df (pandas.DataFrame): Input sales data
        
        Returns:
        pandas.DataFrame: Preprocessed data
        """
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract additional time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Create lag features
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        
        # Rolling window features
        df['sales_rolling_mean_7'] = df['sales'].rolling(window=7).mean()
        df['sales_rolling_std_7'] = df['sales'].rolling(window=7).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df

    def prepare_features_and_target(self, df):
        """
        Prepare features and target variable for model training
        
        Parameters:
        df (pandas.DataFrame): Preprocessed sales data
        
        Returns:
        tuple: X (features), y (target)
        """
        # Select features
        feature_columns = [
            'year', 'month', 'day', 'day_of_week', 'quarter', 
            'sales_lag_1', 'sales_lag_7', 
            'sales_rolling_mean_7', 'sales_rolling_std_7'
        ]
        
        X = df[feature_columns]
        y = df['sales']
        
        return X, y

    def split_time_series_data(self, X, y, train_ratio=0.8):
        """
        Split time series data into training and testing sets
        
        Parameters:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable
        train_ratio (float): Proportion of data to use for training
        
        Returns:
        tuple: X_train, X_test, y_train, y_test
        """
        split_index = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        return X_train, X_test, y_train, y_test

    def train_xgboost_model(self, X_train, y_train):
        """
        Train XGBoost regression model for sales forecasting
        
        Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target variable
        
        Returns:
        tuple: Trained model and scaler
        """
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Define XGBoost model parameters
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Create and train the model
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Store model and scaler
        self.model = model
        self.scaler = scaler
        
        return model, scaler

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        X_test (pandas.DataFrame): Testing features
        y_test (pandas.Series): Testing target variable
        
        Returns:
        tuple: Performance metrics and predictions
        """
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate performance metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred

def main():
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="E-Commerce Sales Forecasting",
        page_icon="📊",
        layout="wide"
    )

    # Create forecasting instance
    forecaster = ECommerceForecasting()

    # Main title
    st.title("🚀 AI-Powered E-Commerce Sales Forecasting")
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose your view",
        [
            "Data Overview", 
            "Model Training", 
            "Sales Forecast", 
            "Performance Metrics"
        ]
    )

    # Load sample data (cached for performance)
    @st.cache_data
    def load_cached_data():
        return forecaster.load_sample_data()

    data = load_cached_data()

    if app_mode == "Data Overview":
        st.header("📈 Sales Data Overview")
        
        # Basic data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sales", f"${data['sales'].sum():,.2f}")
        with col2:
            st.metric("Average Daily Sales", f"${data['sales'].mean():,.2f}")
        with col3:
            st.metric("Sales Volatility", f"${data['sales'].std():,.2f}")
        
        # Sales trend visualization
        fig = px.line(
            data, 
            x='date', 
            y='sales', 
            title='Daily Sales Trend',
            labels={'sales': 'Sales', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "Model Training":
        st.header("🤖 Model Training")
        
        # Preprocess data
        preprocessed_data = forecaster.preprocess_data(data)
        
        # Prepare features
        X, y = forecaster.prepare_features_and_target(preprocessed_data)
        
        # Split data
        X_train, X_test, y_train, y_test = forecaster.split_time_series_data(X, y)
        
        # Train model
        model, scaler = forecaster.train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        metrics, predictions = forecaster.evaluate_model(X_test, y_test)
        
        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"${metrics['MAE']:,.2f}")
        with col2:
            st.metric("MSE", f"${metrics['MSE']:,.2f}")
        with col3:
            st.metric("RMSE", f"${metrics['RMSE']:,.2f}")
        with col4:
            st.metric("R² Score", f"{metrics['R2']:.2%}")

        # Predictions vs Actual Plot
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions
        })
        fig = px.line(
            comparison_df, 
            title='Actual vs Predicted Sales',
            labels={'value': 'Sales', 'variable': 'Type'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "Sales Forecast":
        st.header("🔮 Future Sales Forecast")
        
        # Forecast inputs
        st.sidebar.header("Forecast Parameters")
        forecast_days = st.sidebar.slider(
            "Number of Days to Forecast", 
            min_value=7, 
            max_value=90, 
            value=30
        )
        
        # Simple forecast projection
        last_data_point = data['sales'].iloc[-1]
        forecast_dates = pd.date_range(
            start=data['date'].max() + pd.Timedelta(days=1), 
            periods=forecast_days
        )
        
        # Create forecast with some variability
        forecast_values = [
            last_data_point * (1 + np.random.normal(0, 0.05)) 
            for _ in range(forecast_days)
        ]
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecasted_sales': forecast_values
        })
        
        # Visualize forecast
        fig = px.line(
            forecast_df, 
            x='date', 
            y='forecasted_sales', 
            title='Sales Forecast',
            labels={'forecasted_sales': 'Predicted Sales', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif app_mode == "Performance Metrics":
        st.header("📊 Detailed Performance Analysis")
        st.write("Additional performance metrics and visualizations would be added here.")

if __name__ == "__main__":
    main()