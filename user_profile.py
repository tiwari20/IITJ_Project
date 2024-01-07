import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dataset_processing import DataSetProcessing
from data_preprocessing import DataPreProcessing

class UserProfile:
    stopwords_list = stopwords.words('english') + stopwords.words('portuguese') 
    def __init__(self):
        self.dataset_processing = DataSetProcessing()
        self.data_pre_processing = DataPreProcessing()
        self.raw_news_dataset=self.dataset_processing.read_news_dataset()
        self.user_full_intercation_df=self.data_pre_processing.get_full_user_ineteraction_with_log_trasnformation()
        self.interactions_train_df, self.interactions_test_df= self.data_pre_processing.get_train_test_split_user_interaction_dataset(self.user_full_intercation_df)
        
        self.get_tfidf_matrix()
        
    def get_tfidf_matrix(self):
        vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=self.stopwords_list)

        self.item_ids = self.raw_news_dataset['contentId'].tolist()
        self.tfidf_matrix = vectorizer.fit_transform(self.raw_news_dataset['title'] + "" + self.raw_news_dataset['text'])
        self.tfidf_feature_names = vectorizer.get_feature_names_out()
        self.build_users_profiles()
        return self.tfidf_matrix
    def get_item_profile(self,item_id):
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx+1]
        return item_profile

    def get_item_profiles(self,ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_users_profile(self,person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_item_profiles = self.get_item_profiles(interactions_person_df['contentId'])
        
        user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
        #Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = normalize(np.asarray(user_item_strengths_weighted_avg))
        return user_profile_norm

    def build_users_profiles(self): 
        interactions_indexed_df = self.interactions_train_df[self.interactions_train_df['contentId'] \
                                                    .isin(self.raw_news_dataset['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = self.build_users_profile(person_id, interactions_indexed_df)
        self.user_profiles=user_profiles
        return user_profiles
    
    #return all user profiles
    def get_user_profiles(self):
        user_profiles = self.user_profiles
        return user_profiles
    def get_user_profile_by_id(self, id=-1479311724257856983):
        user_profiles = self.get_user_profiles()
        myprofile = user_profiles[id]
        print(myprofile.shape)
        df=pd.DataFrame(sorted(zip(self.tfidf_feature_names, 
                        user_profiles[-1479311724257856983].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])
        return df
    # Collaborative filtering 
    def get_sparse_pivot_table_users(self):
        #Creating a sparse pivot table with users in rows and items in columns
        self.users_items_pivot_matrix_df = self.interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)
        self.users_items_pivot_matrix = self.users_items_pivot_matrix_df.values
        self.users_items_pivot_sparse_matrix = csr_matrix(self.users_items_pivot_matrix)
