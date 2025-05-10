import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class VolunteerRecommender:
    def __init__(self, supabase, jobs_table='jobs'):
        self.supabase = supabase
        self.jobs_table = jobs_table
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def fetch_data(self):
        response = self.supabase.table(self.jobs_table).select(
            'id, title, organization, description, date, location, skills, requirement'
        ).execute()
        self.df = pd.DataFrame(response.data)
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['organization'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['date'].fillna('') + ' ' +
            self.df['location'].fillna('') + ' ' +
            self.df['skills'].fillna('') + ' ' +
            self.df['requirement'].fillna('')
        )

    def fit(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])

    def build_user_profile(self, user_id):
        # fetch jobs
        response = self.supabase.table('user_interests').select('job_id').eq('user_id', user_id).execute()
        job_ids = [row['job_id'] for row in response.data]
        if not job_ids:
            raise ValueError("User has no interests/jobs associated.")

        # fetch opportunities
        opportunities = self.supabase.table(self.jobs_table).select(
            'title,organization,description,location,skills,requirement'
        ).in_('id', job_ids).execute()
        opportunity_details = opportunities.data

        user_text = " ".join([
            f"{opp.get('title', '')} {opp.get('organization', '')} {opp.get('description', '')} "
            f"{opp.get('location', '')} {opp.get('skills', '')} {opp.get('requirement', '')}"
            for opp in opportunity_details
        ])
        return user_text

    def recommend_for_user(self, user_id, top_n=5):
        # return datafram
        user_text = self.build_user_profile(user_id)
        user_vec = self.vectorizer.transform([user_text])
        cosine_similarities = linear_kernel(user_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity'] = cosine_similarities[top_indices]
        return recommendations[['id', 'title', 'description', 'similarity']]
    
    def paragraph_process(self, query_text, top_n=5):
        query_vec = self.vectorizer.transform([query_text])
        cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity'] = cosine_similarities[top_indices]
        return recommendations[['id', 'title', 'description', 'similarity']]

