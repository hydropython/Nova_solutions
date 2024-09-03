import os
import pandas as pd
import talib as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockAnalysis:
    def __init__(self, folder_path):
        """Initialize the class by specifying the folder path containing CSV files."""
        self.folder_path = folder_path
        self.file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        print(f"Files found: {self.file_names}")

    def load_data(self, file_name):
        """Loads a single CSV file from the folder path and prints time range.
        Also plots raw data in subplots."""
        temp_df = pd.read_csv(os.path.join(self.folder_path, file_name))
        temp_df['File'] = file_name  # Add a column for the file name
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        temp_df.set_index('Date', inplace=True)

        # Print the time range
        start_date = temp_df.index.min()
        end_date = temp_df.index.max()
        print(f"Data loaded for {file_name}. Time range: {start_date} to {end_date}")

        # Plot raw data
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Open Price', 'Close Price',
            'High Price', 'Low Price',
            'Volume'
        ))

        # Plot Open Price
        fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df['Open'], mode='lines', name='Open Price', line=dict(color='blue')), row=1, col=1)

        # Plot Close Price
        fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df['Close'], mode='lines', name='Close Price', line=dict(color='darkgreen')), row=1, col=2)

        # Plot High Price
        fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df['High'], mode='lines', name='High Price', line=dict(color='red')), row=2, col=1)

        # Plot Low Price
        fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df['Low'], mode='lines', name='Low Price', line=dict(color='orange')), row=2, col=2)

        fig.update_layout(title=f'Raw Data ({file_name})', template='plotly_white', height=800)  # Updated to white background
        fig.show()

        return temp_df

    def ensure_columns(self, df):
        """Ensure the DataFrame contains the required columns."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns in data - {', '.join(missing_columns)}")
        else:
            print("All required columns are present.")
        return df

    def calculate_technical_indicators(self, df):
        """Calculates various technical indicators using TA-Lib."""
        # Ensure necessary columns are present
        df = self.ensure_columns(df)
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

        # MACD (Moving Average Convergence Divergence)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # ATR (Average True Range)
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        print(f"Technical indicators calculated for {df['File'].iloc[0]}.")
        print(df[['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower']].head())
        return df

    def calculate_moving_averages(self, df, periods=[20, 50]):
        """Calculates various moving averages for the specified periods."""
        for period in periods:
            df[f'SMA{period}'] = ta.SMA(df['Close'], timeperiod=period)
            df[f'EMA{period}'] = ta.EMA(df['Close'], timeperiod=period)
        print(f"Moving averages calculated for {df['File'].iloc[0]}.")
        print(df[[f'SMA{period}' for period in periods] + [f'EMA{period}' for period in periods]].head())
        return df

    def portfolio_performance(self, portfolio_weights, all_dfs):
        """Calculates and visualizes portfolio performance based on given weights."""
        portfolio_returns = pd.DataFrame()
        for df in all_dfs:
            file_name = df['File'].iloc[0]
            df['Daily_Return'] = df['Close'].pct_change()
            df['Weighted_Return'] = df['Daily_Return'] * portfolio_weights.get(file_name, 0)
            portfolio_returns[file_name] = df['Weighted_Return']

        # Calculate portfolio return
        portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)
        print("Portfolio performance calculated.")
        print(portfolio_returns.head())
        return portfolio_returns

    def display_portfolio_weights(self, portfolio_weights):
        """Displays portfolio weights in a bar chart."""
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(portfolio_weights.keys()), y=list(portfolio_weights.values()), marker_color='lightblue'))
        fig.update_layout(title='Portfolio Weights', xaxis_title='Stock', yaxis_title='Weight', template='plotly_white')  # Updated to white background
        fig.show()
        print("Portfolio weights displayed.")

    def visualize_data(self, df):
        """Visualizes various technical indicators in separate plots."""
        # Plot closing prices along with moving averages
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='darkgreen')))
        for col in df.columns:
            if col.startswith('SMA') or col.startswith('EMA'):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(title=f'Stock Price and Moving Averages ({df["File"].iloc[0]})', template='plotly_white')  # Updated to white background
        fig.show()

        # Plot ATR
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(color='purple')))
        fig.update_layout(title=f'ATR ({df["File"].iloc[0]})', template='plotly_white')  # Updated to white background
        fig.show()

        # Plot RSI
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='yellow')))
        fig.add_shape(type='line', x0=df.index.min(), y0=70, x1=df.index.max(), y1=70, line=dict(color='red', dash='dash'))
        fig.add_shape(type='line', x0=df.index.min(), y0=30, x1=df.index.max(), y1=30, line=dict(color='red', dash='dash'))
        fig.update_layout(title=f'RSI ({df["File"].iloc[0]})', template='plotly_white')  # Updated to white background
        fig.show()

        # Plot MACD
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color='grey'))
        fig.update_layout(title=f'MACD ({df["File"].iloc[0]})', template='plotly_white')  # Updated to white background
        fig.show()

        # Plot Bollinger Bands
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='darkgreen')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], mode='lines', name='BB Middle', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(color='red')))
        fig.update_layout(title=f'Bollinger Bands ({df["File"].iloc[0]})', template='plotly_white')  # Updated to white background
        fig.show()

    
    def run_analysis(self, file_name, portfolio_weights=None):
        """Runs the full analysis pipeline for a specific file."""
        df = self.load_data(file_name)
        df = self.calculate_technical_indicators(df)
        df = self.calculate_moving_averages(df)

        self.visualize_data(df)
        
        if portfolio_weights:
            # Load data for all files and calculate portfolio performance
            all_dfs = [self.load_data(f) for f in self.file_names]
            all_dfs = [self.ensure_columns(df) for df in all_dfs]  # Ensure columns are present
            all_dfs = [self.calculate_technical_indicators(df) for df in all_dfs]
            all_dfs = [self.calculate_moving_averages(df) for df in all_dfs]
            portfolio_returns = self.portfolio_performance(portfolio_weights, all_dfs)
            self.display_portfolio_weights(portfolio_weights)
            return portfolio_returns