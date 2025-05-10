'''
recommender = VolunteerRecommender(supabase)
recommender.fetch_data()
recommender.fit()
user_id = "blank"
user_embedding = recommender.build_user_profile(user_id)
recommendations = recommender.recommend_for_user(user_embedding, top_n=5)
print(recommendations)
'''

'''
PARAGRAPH PROCESSING

query = "I am free on weekends and want to volunteer with kids."
recommendations = recommender.paragraph_process(query, top_n=5)
print(recommendations)



'''


'''
recommender = llmRecommender(supabase)
recommender.fetch_data()
recommender.load_data()
recommender.build_qa_chain()
recommendations = recommender.recommend(blurb)
print(recommendations)

pip freeze > requirements.txt
'''
