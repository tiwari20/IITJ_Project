from  recommender_data_helper import RecommenderDataHelper
from data_preprocessing import DataPreProcessing
from dataset_processing import DataSetProcessing
import pandas as pd
import random
class TopNModelEvaluator:
    EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
    
    def __init__(self):
        self.dataset_processing=DataSetProcessing()
        self.reccomender_data_helper=RecommenderDataHelper()
        self.data_pre_processing=DataPreProcessing()
        self.load_indexed_data()
        

    def load_indexed_data(self):
        self.user_full_intercation_df=self.data_pre_processing.get_full_user_ineteraction_with_log_trasnformation()
        interactions_train_df, interactions_test_df= self.data_pre_processing.get_train_test_split_user_interaction_dataset(self.user_full_intercation_df)
        self.user_full_intercation_df_indexed,self.interactions_train_df_indexed, self.interactions_test_df_indexed = self.reccomender_data_helper.indexing_datasets_by_person_id(self.user_full_intercation_df,interactions_train_df,interactions_test_df)
        self.raw_full_news_dataset=self.dataset_processing.read_news_dataset()
        
    def get_items_not_interacted(self, person_id, sample_size, seed=42):
        interacted_items =  self.reccomender_data_helper.get_items_interacted(person_id, self.user_full_intercation_df_indexed)
        all_items = set(self.raw_full_news_dataset['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def user_wise_evaluate(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = self.interactions_test_df_indexed.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_news(person_id, 
                                               items_to_ignore= self.reccomender_data_helper.get_items_interacted(person_id, 
                                                                                    self.interactions_train_df_indexed), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_items_not_interacted(person_id, 
                                                                          sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_result(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(self.interactions_test_df_indexed.index.unique().values)):
            person_metrics = self.user_wise_evaluate(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_name_of_model(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    