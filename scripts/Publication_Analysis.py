import pandas as pd
import plotly.express as px

class PublisherAnalysis:
    def __init__(self, data_path):
        # Load the data from the CSV file
        self.data = pd.read_csv(data_path)

    def extract_domains(self):
        if isinstance(self.data['publisher'], pd.Series):
            print("Publisher column is a pandas Series")

            if all(isinstance(item, str) for item in self.data['publisher']):
                self.data['domain'] = self.data['publisher'].str.split('@').str[-1]
            else:
                print("Error: Not all entries in 'publisher' are strings.")
        else:
            print("Error: 'publisher' is not a Series. Check your data format.")
        
    def plot_top_publishers(self):
        publisher_counts = self.data['publisher'].value_counts()
        top_publishers = publisher_counts.head(10)
        
        fig = px.bar(
            top_publishers,
            x=top_publishers.index,
            y=top_publishers.values,
            labels={'x': 'Publisher', 'y': 'Number of Publications'},
            title='Top 10 Publishers by Number of Publications'
        )
        
        fig.update_layout(template='plotly_white')
        fig.show()

    def plot_domain_frequencies(self):
        domain_counts = self.data['domain'].value_counts()
        
        fig = px.bar(
            domain_counts,
            x=domain_counts.index,
            y=domain_counts.values,
            labels={'x': 'Domain', 'y': 'Number of Publications'},
            title='Frequency of Domains in Publisher Email Addresses'
        )
        
        fig.update_layout(template='plotly_white')
        fig.show()

    def plot_publication_trends(self):
        # Try parsing dates with error handling
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        except Exception as e:
            print("Error parsing dates:", e)
            # Display problematic entries
            print(self.data['date'].head(10))
        
        # Drop rows where 'date' could not be parsed
        self.data = self.data.dropna(subset=['date'])
        
        # Count publications by date
        daily_publications = self.data.groupby(self.data['date'].dt.date).size()
        
        fig = px.line(
            daily_publications,
            x=daily_publications.index,
            y=daily_publications.values,
            labels={'x': 'Date', 'y': 'Number of Publications'},
            title='Daily Publication Trends'
        )
        
        fig.update_layout(template='plotly_white')
        fig.show()

    def analyze_unique_domains(self):
        self.extract_domains()  # Ensure domains are extracted
        # Count frequencies of unique domains
        domain_counts = self.data['domain'].value_counts()
        # Display the top 10 domains
        print(domain_counts.head(10))
        # Plot the domain frequencies
        self.plot_domain_frequencies()
