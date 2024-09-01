import pandas as pd
import plotly.express as px
import os

class StockCorr:
    def __init__(self, folder_path, width=800, height=800):
        """
        Initialize the StockCorr class with the path to the folder containing CSV files.
        
        Parameters:
        folder_path (str): Path to the folder containing CSV files.
        width (int): Width of the plot in pixels.
        height (int): Height of the plot in pixels.
        """
        self.folder_path = folder_path
        self.width = width
        self.height = height
        self.data = self._load_data()
    
    def _load_data(self):
        """
        Load 'Date' and 'Close' columns from all CSV files in the folder and return a DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame with 'Date' and 'Close' columns from all files.
        """
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        data = pd.DataFrame()

        for file in csv_files:
            df = pd.read_csv(os.path.join(self.folder_path, file))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
                file_name = os.path.splitext(file)[0]
                df = df[['Date', 'Close']]
                df.rename(columns={'Close': file_name}, inplace=True)
                data = pd.merge(data, df, on='Date', how='outer') if not data.empty else df
            else:
                print(f"'Date' or 'Close' column not found in {file}")

        return data
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix of the 'Close' prices.
        """
        correlation_matrix = self.data.drop(columns='Date').corr()
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            title='Correlation Matrix of Close Prices',
            color_continuous_scale='Viridis',
            width=self.width,
            height=self.height
        )
        fig.show()
    
    def plot_close_prices(self):
        """
        Plot the 'Close' prices time series for each CSV file.
        """
        combined_df = self.data.melt(id_vars='Date', var_name='file_name', value_name='Close')
        fig = px.line(
            combined_df,
            x='Date',
            y='Close',
            color='file_name',
            title='Close Prices from Combined CSV Files',
            width=self.width,
            height=self.height
        )
        fig.show()
