from fastapi import FastAPI
from fastapi.responses import JSONResponse
app = FastAPI()
from recommender_systems import RecommenderSystems

recommender_systems = RecommenderSystems()

@app.get("/")
async def root():
    return {"message": "ITTJ MTP Project"}

@app.get("/get_news_popularrity_model_TopN_metrics")
async def get_news_popularrity_model_TopN_metrics():
    output =recommender_systems.fetch_result_popularrity_model()
    return JSONResponse(content=output, media_type="application/json")

@app.get("/get_news_content_based_filtering_TopN_metrics")
async def get_news_content_based_filtering_TopN_metrics():
    output =recommender_systems.get_result_content_based_filtering()
    return JSONResponse(content=output, media_type="application/json")

@app.get("/get_collaborative_based_filtering_TopN_metrics")
async def get_collaborative_based_filtering_TopN_metrics():
    output =recommender_systems.get_result_collaborative_based_filtering()
    return JSONResponse(content=output, media_type="application/json")
@app.get("/get_hybrid_based_filtering_TopN_metrics")
async def get_hybrid_based_filtering_TopN_metrics():
    output =recommender_systems.get_result_hybrid_recommender_model()
    return JSONResponse(content=output, media_type="application/json")

# get the interaction of news of the given user
@app.post("/review_user_interactions_from_dataset/{user_id}")
async def review_user_interactions_from_dataset(user_id):
    output =recommender_systems.get_most_interacted_news_by_user(int(user_id),test_set=False)
    return JSONResponse(content=output, media_type="application/json")

#reccomendations based on the different models 

#content filtering based recommeded news for user
@app.post("/content_filtering_recommended_news_for_user/{user_id}")
async def content_filtering_recommended_news_for_user(user_id='-1479311724257856983'):
    output =recommender_systems.content_based_recommender_model.recommend_news(int(user_id),  topn=20, verbose=True)
    return JSONResponse(content=output.to_dict("records"), media_type="application/json")


#collaborative filtering based recommeded news for user
@app.post("/collaborative_filtering_recommended_news_for_user/{user_id}")
async def collaborative_filtering_recommended_news_for_user(user_id='-1479311724257856983'):
    output =recommender_systems.collaborative_filtering_recommender_model.recommend_news(int(user_id),  topn=20, verbose=True)
    return JSONResponse(content=output.to_dict("records"), media_type="application/json")


#hybrid filtering based recommeded news for user
@app.post("/hybrid_recommended_news_for_user/{user_id}")
async def hybrid_recommended_news_for_user(user_id='-1479311724257856983'):
    output =recommender_systems.hybrid_recommender_model.recommend_news(int(user_id),  topn=20, verbose=True)
    return JSONResponse(content=output.to_dict("records"), media_type="application/json")
    


