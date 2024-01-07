from top_n_model_evaluator import TopNModelEvaluator
from recommender_system_popularity import Recommender_System_Popularity
from data_preprocessing import DataPreProcessing
from dataset_processing import DataSetProcessing
from recommender_system_content_filtering import Recommender_System_Content_Filtering
from recommender_system_collaborative_filtering import Recommender_System_Collaborative_Filtering
from matrix_factorization import MatrixFactorization
from recommender_system_hybrid import HybridRecommender
from recommender_data_helper import RecommenderDataHelper


class RecommenderSystems:
    def __init__(self):
        self.data_pre_processing = DataPreProcessing()
        self.dataset_processing = DataSetProcessing()
        self.recommender_data_helper= RecommenderDataHelper()
        
        self.top_n_evaluator = TopNModelEvaluator()
        self.interactions_full_df = self.data_pre_processing.get_full_user_ineteraction_with_log_trasnformation()
        self.raw_news_dataset =self.dataset_processing.read_news_dataset()
        self.user_full_intercation_df=self.data_pre_processing.get_full_user_ineteraction_with_log_trasnformation()
        
        #train and test sets
        self.interactions_train_df, self.interactions_test_df= self.data_pre_processing.get_train_test_split_user_interaction_dataset(self.user_full_intercation_df)
        self.interactions_full_indexed_df,self.interactions_train_df_indexed, self.interactions_test_df_indexed =self.recommender_data_helper.indexing_datasets_by_person_id(self.user_full_intercation_df,self.interactions_train_df, self.interactions_test_df)
        #get most popular items
        self.get_most_popular_items()
        
        #initiating content based model
        self.content_based_recommender_model = Recommender_System_Content_Filtering(self.raw_news_dataset)
        
        #intiating collaborative based model 
        matrixfactorization = MatrixFactorization(self.interactions_train_df)
        self.collaborative_filtering_recommender_model = Recommender_System_Collaborative_Filtering(matrixfactorization.cf_preds_df,self.raw_news_dataset)
        
        #Initiating hybrid based recommender model
        self.hybrid_recommender_model = HybridRecommender(self.content_based_recommender_model, self.collaborative_filtering_recommender_model, self.raw_news_dataset,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=100.0)
        
        
         
              
        
        
    def get_most_popular_items(self):
        
        #Computes the most popular items
        self.news_dataframe_popularity = self.interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
       
       
    # Recall@5 is 0.2417 which means that about **24%** of interacted items in test set were ranked by Popularity model among the top-5 items
    # Recall@10 is 37%
    def fetch_result_popularrity_model(self):
        self.get_most_popular_items()
        popularity_model = Recommender_System_Popularity(self.news_dataframe_popularity, self.raw_news_dataset)
        
        print('Evaluating Popularity recommendation model...')
        pop_global_metrics, pop_detailed_results_df = self.top_n_evaluator.evaluate_result(popularity_model)
        print('\nGlobal metrics:\n%s' % pop_global_metrics)
        pop_detailed_results_df_dict=pop_detailed_results_df.head(10).to_dict("records")
        return {'Global metrics': pop_global_metrics, "result":pop_detailed_results_df_dict }
        # return pop_detailed_results_df.head(10)
        
    #Recall@5 isabout **0.162**, which means that about **16%** of interacted items in test set were ranked by this model among the top-5 items (from lists with 100 random items).
    # And **Recall@10** was **0.261 (52%)**.
    def get_result_content_based_filtering(self):
        print('Evaluating Content-Based Filtering model...')
        cb_global_metrics, cb_detailed_results_df =self.top_n_evaluator.evaluate_result(self.content_based_recommender_model)
        print('\nGlobal metrics:\n%s' % cb_global_metrics)
        print(cb_detailed_results_df.head(10))
        pop_detailed_results_df_dict=cb_detailed_results_df.head(10).to_dict("records")
        return {'Global metrics': cb_global_metrics, "result":pop_detailed_results_df_dict }
    
        
    def get_result_collaborative_based_filtering(self):
        # matrixfactorization = MatrixFactorization(self.interactions_train_df)
        # self.collaborative_filtering_recommender_model = Recommender_System_Collaborative_Filtering(matrixfactorization.cf_preds_df,self.raw_news_dataset)
        
        # > Evaluating the Collaborative Filtering model (SVD matrix factorization), we observe that we got **Recall@5 (33%)** and **Recall@10 (46%)** values, 
        print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
        cf_global_metrics, cf_detailed_results_df = self.top_n_evaluator.evaluate_result(self.collaborative_filtering_recommender_model)
        print('\nGlobal metrics:\n%s' % cf_global_metrics)
        print(cf_detailed_results_df.head(10))
        pop_detailed_results_df_dict=cf_detailed_results_df.head(10).to_dict("records")
        return {'Global metrics': cf_global_metrics, "result":pop_detailed_results_df_dict }
        
    def get_result_hybrid_recommender_model(self):
       
        print('Evaluating Hybrid model...')
        hybrid_global_metrics, hybrid_detailed_results_df = self.top_n_evaluator.evaluate_result(self.hybrid_recommender_model)
        print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
        print(hybrid_detailed_results_df.head(10))
        pop_detailed_results_df_dict=hybrid_detailed_results_df.head(10).to_dict("records")
        return {'Global metrics': hybrid_global_metrics, "result":pop_detailed_results_df_dict }
    
    #get most interacted news by user
    def get_most_interacted_news_by_user(self,person_id, test_set=True):
        if test_set:
            interactions_df = self.interactions_test_df_indexed
        else:
            interactions_df = self.interactions_train_df_indexed
        output= interactions_df.loc[person_id].merge(self.raw_news_dataset, how = 'left', 
                                                        left_on = 'contentId', 
                                                        right_on = 'contentId') \
                            .sort_values('eventStrength', ascending = False)[['eventStrength', 
                                                                            'contentId',
                                                                            'title', 'url', 'lang']]
        return output.head(20).to_dict("records")
        
    
