import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VolunteerRecommender:
    def __init__(self, supabase, jobs_table='jobs'):
        self.supabase = supabase
        self.jobs_table = jobs_table
        self.df = None
        self.keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')
        self.embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.embeddings = None

    def fetch_data(self):
        # fetch opportunities
        response = self.supabase.table(self.jobs_table).select(
            'id, title, organization, description, date, location, skills, requirement'
        ).execute()
        self.df = pd.DataFrame(response.data)
        self.preprocess()

    def preprocess(self):
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['organization'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['location'].fillna('') + ' ' +
            self.df['skills'].fillna('') + ' ' +
            self.df['requirement'].fillna('')
    )

    def fit(self):
        # print("MODEL IS CALLED")
        # embed
        self.embeddings = self.embedding_model.encode(
            self.df['combined_text'].tolist(), convert_to_numpy=True
        )

    def build_user_profile(self, user_id):
        # fetch job id
        response = self.supabase.table('user_interests').select('job_id').eq('user_id', user_id).execute()
        job_ids = [row['job_id'] for row in response.data]
        if not job_ids:
            raise ValueError("User has no interests/jobs associated.")
        
        # fetch opportunity details
        opportunities = self.supabase.table(self.jobs_table).select(
            'title,organization,description,location,skills,requirement'
        ).in_('id', job_ids).execute()
        opportunity_details = opportunities.data
        
        # process
        user_text = " ".join([
            f"{opp.get('title', '')} {opp.get('organization', '')} {opp.get('description', '')} "
            f"{opp.get('location', '')} {opp.get('skills', '')} {opp.get('requirement', '')}"
            for opp in opportunity_details
        ])
        
        # generate user embedding
        user_embedding = self.embedding_model.encode([user_text], convert_to_numpy=True)
        return user_embedding

    def recommend_for_user(self, user_embedding, top_n=5):
        # return dataframe
        similarities = cosine_similarity(user_embedding, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity'] = similarities[top_indices]
        return recommendations
    
    def paragraph_process(self, query_text, top_n=5):
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]

        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity'] = similarities[top_indices]
        return recommendations[['id', 'title', 'description', 'similarity']]

