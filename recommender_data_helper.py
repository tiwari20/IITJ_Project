import pandas as pd

class RecommenderDataHelper: 
    #Indexing by personId to speed up the searches during evaluation
    def indexing_datasets_by_person_id(self,interactions_full_df,interactions_train_df, interactions_test_df ):
        interactions_full_indexed_df = interactions_full_df.set_index('personId')
        interactions_train_indexed_df = interactions_train_df.set_index('personId')
        interactions_test_indexed_df = interactions_test_df.set_index('personId')
        return interactions_full_indexed_df,interactions_train_indexed_df,interactions_test_indexed_df
    
    # Get the user's data and merge in the News information.
    def get_items_interacted(self,person_id, interactions_df):
        
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    