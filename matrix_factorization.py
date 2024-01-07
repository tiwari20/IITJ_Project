import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class MatrixFactorization:
    def __init__(self,interactions_train_df):
        self.interactions_train_df=interactions_train_df
        self.get_sparse_pivot_table_users()
        self.performMF()
        
    def get_sparse_pivot_table_users(self):
        #Creating a sparse pivot table with users in rows and items in columns
        self.users_items_pivot_matrix_df = self.interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)
        self.users_items_pivot_matrix = self.users_items_pivot_matrix_df.values
        
        self.users_ids = list(self.users_items_pivot_matrix_df.index)
        
        self.users_items_pivot_sparse_matrix = csr_matrix(self.users_items_pivot_matrix)
        
        
    def performMF(self):
        #The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 15
        #Performs matrix factorization of the original user item matrix
        
        U, sigma, Vt = svds(self.users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        
        # After the factorization, we try to to reconstruct the original matrix by multiplying its factors. The resulting matrix is not sparse any more. It was generated predictions for items the user have not yet interaction, which we will exploit for recommendations.
        self.all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
        self.all_user_predicted_ratings_norm = (self.all_user_predicted_ratings - self.all_user_predicted_ratings.min()) / (self.all_user_predicted_ratings.max() - self.all_user_predicted_ratings.min())
        #Converting the reconstructed matrix back to a Pandas dataframe
        self.cf_preds_df = pd.DataFrame(self.all_user_predicted_ratings_norm, columns = self.users_items_pivot_matrix_df.columns, index=self.users_ids).transpose()
    

