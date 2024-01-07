import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from user_profile import UserProfile

class Recommender_System_Content_Filtering:
    
    approach_for_recommender = 'Content-Based Filtering model'
    
    def __init__(self, items_df=None):
        self.user_profile = UserProfile()
        self.item_ids = self.user_profile.item_ids
        self.items_df = items_df
        
    def get_name_of_model(self):
        return self.approach_for_recommender
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profile.user_profiles[person_id],self.user_profile.tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.user_profile.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_news(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        output_datarframe_recommendations = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            output_datarframe_recommendations = output_datarframe_recommendations.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return output_datarframe_recommendations