
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class EDA_Analytics:
    def __init__(self, file_path):
        """Initialize with the path to the CSV file."""
        self.df = pd.read_csv(file_path)

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame and handle them.
        """
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)
        return missing_values

    def fill_missing_values(self, strategy='mean'):
        """
        Fill missing values in the DataFrame based on the specified strategy.
        """
        if strategy == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif strategy == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        elif strategy == 'none':
            print("No filling strategy applied. Missing values remain as is.")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'none'.")
        
        print(f"Missing values filled using strategy: {strategy}")
        return self.df

    def compute_headline_length_stats(self):
        """Compute basic statistics for headline lengths."""
        self.df['headline_length'] = self.df['headline'].apply(len)
        headline_stats = self.df['headline_length'].describe()
        print("Headline Length Statistics:")
        print(headline_stats)
        
        fig = px.histogram(
            self.df, 
            x='headline_length', 
            nbins=30, 
            title='Distribution of Headline Lengths',
            color_discrete_sequence=['teal'], 
            labels={'headline_length': 'Length (characters)'}
        )
        fig.update_layout(
            title=dict(text='Distribution of Headline Lengths', font=dict(size=30, family='Arial, sans-serif', color='#333333')),
            xaxis_title='Length (characters)', 
            yaxis_title='Frequency', 
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18))
        )
        fig.show()
    
    def count_articles_per_publisher(self):
        """Count the number of articles per publisher and visualize the results."""
        articles_per_publisher = self.df['publisher'].value_counts()
        print("Number of Articles per Publisher:")
        print(articles_per_publisher)
        
        fig = px.bar(
            articles_per_publisher, 
            x=articles_per_publisher.index, 
            y=articles_per_publisher.values,
            title='Number of Articles per Publisher', 
            labels={'x': 'Publisher', 'y': 'Number of Articles'},
            color_discrete_sequence=['teal']
        )
        fig.update_layout(
            title=dict(text='Number of Articles per Publisher', font=dict(size=30, family='Arial, sans-serif', color='#333333')),
            xaxis_title='Publisher', 
            yaxis_title='Number of Articles', 
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18))
        )
        fig.show()
    
    def analyze_publication_dates(self):
        """Analyze publication dates for trends and visualize the results."""
        try:
            self.df['date'] = pd.to_datetime(self.df['date'], format=None, infer_datetime_format=True, errors='coerce')
        except Exception as e:
            print(f"Date conversion error: {e}")
            print("Check the format of the date strings in your dataset.")
            return

        if self.df['date'].isnull().any():
            print("Some dates could not be converted and have been set to NaT (Not a Time).")

        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
        articles_per_day = self.df['day_of_week'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        print("Number of Articles per Day of the Week:")
        print(articles_per_day)
        
        fig = px.bar(
            articles_per_day, 
            x=articles_per_day.index, 
            y=articles_per_day.values,
            title='Number of Articles per Day of the Week', 
            labels={'x': 'Day of the Week', 'y': 'Number of Articles'},
            color_discrete_sequence=['teal']
        )
        fig.update_layout(
            title=dict(text='Number of Articles per Day of the Week', font=dict(size=30, family='Arial, sans-serif', color='#333333')),
            xaxis_title='Day of the Week', 
            yaxis_title='Number of Articles',
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18))
        )
        fig.show()
        
        articles_per_month = self.df['date'].dt.to_period('M').value_counts().sort_index()
        print("Number of Articles per Month:")
        print(articles_per_month)
        
        fig = px.line(
            x=articles_per_month.index.astype(str), 
            y=articles_per_month.values, 
            title='Number of Articles Over Time', 
            labels={'x': 'Month', 'y': 'Number of Articles'},
            line_shape='linear', 
            color_discrete_sequence=['teal']
        )
        fig.update_layout(
            title=dict(text='Number of Articles Over Time', font=dict(size=30, family='Arial, sans-serif', color='#333333')),
            xaxis_title='Month', 
            yaxis_title='Number of Articles',
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=18))
        )
        fig.show()

    def perform_nltk_sentiment_analysis(self):
        """
        Perform sentiment analysis on headlines using NLTK's TextBlob.
        """
        # Ensure 'headline' column exists
        if 'headline' not in self.df.columns:
            raise ValueError("The DataFrame must contain a 'headline' column.")

        def get_sentiment(text):
            analysis = TextBlob(text)
            # Classify sentiment
            if analysis.sentiment.polarity > 0:
                return 'Positive'
            elif analysis.sentiment.polarity < 0:
                return 'Negative'
            else:
                return 'Neutral'

        self.df['nltk_sentiment'] = self.df['headline'].apply(get_sentiment)
        nltk_sentiment_counts = self.df['nltk_sentiment'].value_counts()
        print("NLTK Sentiment Analysis Results:")
        print(nltk_sentiment_counts)
        
        # Plot NLTK sentiment analysis results with luxurious styling
        fig = px.pie(
            nltk_sentiment_counts, 
            values=nltk_sentiment_counts.values, 
            names=nltk_sentiment_counts.index,
            title='NLTK Sentiment Distribution',
            color_discrete_sequence=['#4CAF50', '#FFC107', '#9E9E9E']  # Rich color scheme
        )
        
        fig.update_layout(
            title=dict(text='NLTK Sentiment Distribution', font=dict(size=30, family='Arial, sans-serif', color='#333333')),
            plot_bgcolor='white',  # White background
            paper_bgcolor='white',  # White background
            legend_title_font=dict(size=20, family='Arial, sans-serif', color='#333333'),
            legend=dict(
                title='Sentiment',
                title_font=dict(size=20, family='Arial, sans-serif', color='#333333'),
                font=dict(size=18, family='Arial, sans-serif', color='#333333')
            ),
            margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better spacing
            annotations=[
                dict(
                    text='NLTK Sentiment Distribution',
                    x=0.5,
                    y=0.5,
                    font_size=22,
                    showarrow=False
                )
            ]
        )
        
        fig.update_traces(
            textinfo='percent+label',  # Show percentage and label
            textfont_size=20,
            pull=[0.1, 0.1, 0.1]  # Slightly pull out each slice for emphasis
        )
        
        fig.show()

    def perform_topic_modeling(self, num_topics=5):
        """
        Perform topic modeling using Latent Dirichlet Allocation (LDA).
        """
        headlines = self.df['headline'].dropna().tolist()
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(headlines)
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        
        print("Top words per topic:")
        words = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            print(f"Topic {topic_idx + 1}:")
            print([words[i] for i in topic.argsort()[-10:]])