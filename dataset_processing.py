import opendatasets
import pandas as pd


class DataSetProcessing(object):
    
    
    
    dataset_news_file_name='dataset/shared_articles.csv'
    
    user_interactions_dataset_file_name='dataset/users_interactions.csv'
    
    # def __init__(self):
    #     self.download_dataset()
        
    # def download_dataset(self):
    #     opendatasets.download(self.dataset_url)


    def read_news_dataset(self):
        
        news_dataframe = pd.read_csv(self.dataset_news_file_name)
        news_dataframe = news_dataframe[news_dataframe['eventType'] == 'CONTENT SHARED']
        return news_dataframe

    #Loading the user interaction data on articles, it has the user interaction events like **VIEW**: The user has opened the article. LIKE, COMMENT CREATED, FOLLOW, BOOKMARK"
    def read_users_interactions_dataset(self):
        interactions_df = pd.read_csv(self.user_interactions_dataset_file_name)
        return interactions_df
