import pandas as pd
from dataset_processing import DataSetProcessing
import math
from sklearn.model_selection import train_test_split
news_event_type_strength = { 'VIEW': 1.0,
    'LIKE': 2.0, 
    'BOOKMARK': 2.5, 
    'FOLLOW': 4.0,
    'COMMENT CREATED': 4.0,  
    }
class DataPreProcessing:

    #Associtating the wieghts or strenghts for different user interaction type 
    news_dataset=None
    users_interactions_dataset=None
    
    
    def __init__(self) -> None:
        self.load_news_dataset()
    
    def load_news_dataset(self):
        data_processing= DataSetProcessing()
        self.news_dataset=data_processing.read_news_dataset()
        self.users_interactions_dataset=data_processing.read_users_interactions_dataset()
        print(f"data is loaded and count of news datasets = {len(self.news_dataset)} and user interaction ={len(self.users_interactions_dataset)}")
        
        
    def get_news_event_type_strength(self):
        return self.news_event_type_strength
    
    def get_user_interacation_associated_event_strength(self):
        #associating interacation strength to each integration 
        self.users_interactions_dataset['eventStrength'] = self.users_interactions_dataset['eventType'].apply(lambda x: news_event_type_strength[x])
        return self.users_interactions_dataset
   
   
    # to address the cold start problem, we are keeping the news with atleast 5 interaction 
    #Aggregate all the interactions the user has performed in an item by a weighted sum of interaction type strength
    def get_users_interaction_with_required_count(self,min_num_interaction=5):
       associated_event_strength_dataset=self.get_user_interacation_associated_event_strength()
       users_interactions_count_df = associated_event_strength_dataset.groupby(['personId', 'contentId']).size().groupby('personId').size()
       print('# users: %d' % len(users_interactions_count_df))
       
       users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= min_num_interaction].reset_index()[['personId']]
       print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df)) 
       
       interactions_from_selected_users_df = associated_event_strength_dataset.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
       
       print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
       return interactions_from_selected_users_df


    def smooth_user_preference(self,x):
        return math.log(1+x, 2)
    
    # Apply a log transformation to smooth the distribution.
    def get_full_user_ineteraction_with_log_trasnformation(self):
        
        interactions_from_selected_users_df=self.get_users_interaction_with_required_count(5)
        
        interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(self.smooth_user_preference).reset_index()
        print('# of unique user/item interactions: %d' % len(interactions_full_df))
        return interactions_full_df
    
    # We are using here a simple cross-validation approach named holdout
    def get_train_test_split_user_interaction_dataset(self, interactions_full_df):
        interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)

        print('# interactions on Train set: %d' % len(interactions_train_df))
        print('# interactions on Test set: %d' % len(interactions_test_df))
        return interactions_train_df, interactions_test_df
        
    
