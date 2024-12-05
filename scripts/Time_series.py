import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class PublicationAnalysis:

    def __init__(self, data):
        self.data = data
        self.data['publication_time'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data.set_index('publication_time', inplace=True)
    
    def plot_daily_publications(self):
        daily_counts = self.data.resample('D').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_counts.index, 
            y=daily_counts.values, 
            mode='lines+markers', 
            name='Daily Publications',
            line=dict(color='darkgreen', width=2),
            marker=dict(color='yellow', size=6)
        ))
        
        fig.update_layout(
            title='Daily Publication Frequency',
            xaxis_title='Date',
            yaxis_title='Number of Publications',
            title_font=dict(size=24, color='black'),  # Black title font
            xaxis=dict(tickfont=dict(size=14, color='black')),  # Black axis tick font
            yaxis=dict(tickfont=dict(size=14, color='black')),  # Black axis tick font
            plot_bgcolor='white',  # White plot background
            paper_bgcolor='white'  # White paper background
        )
        
        fig.show()
    
    def plot_publications_by_hour(self):
        self.data['hour'] = self.data.index.hour
        hour_counts = self.data['hour'].value_counts().sort_index()
        
        fig = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            labels={'x': 'Hour of the Day', 'y': 'Number of Publications'},
            title='Publication Distribution by Hour',
            color=hour_counts.values,
            color_continuous_scale=px.colors.sequential.Plasma  # Luxurious color scale
        )
        
        fig.update_layout(
            title_font=dict(size=24, color='black'),  # Black title font
            xaxis=dict(tickfont=dict(size=14, color='black')),  # Black axis tick font
            yaxis=dict(tickfont=dict(size=14, color='black')),  # Black axis tick font
            plot_bgcolor='white',  # White plot background
            paper_bgcolor='white'  # White paper background
        )
        
        fig.show()