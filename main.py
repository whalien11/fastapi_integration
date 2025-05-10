from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase_keybert_recommendation import VolunteerRecommender 
from dotenv import load_dotenv
import os
from supabase import create_client, Client

from llm_recommendation import llmRecommender



app = FastAPI()
load_dotenv()


# declare origin/s
origins = [
    "https://nuvolunteers.org/",
    "localhost:3000"
]


# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase init
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


#jobs
jobs = {}



# Define request schema
class ExampleRequest(BaseModel):
    userid: str
recommender = VolunteerRecommender(supabase)
@app.post("/recommend/")
def recommend(request: ExampleRequest):
    try:
        # Load recommender once

        print("JUST STARTING")
        user_id = request.userid
        print(f"USERID:{user_id}")
        
        print("INIT IS DONE")
        recommender.fetch_data()
        print("Before MODEL FIT")
        recommender.fit()
        print("MODEL FIT IS DONE")
        user_embedding = recommender.build_user_profile(user_id)
        print("USER RECOMMENDATIONS TAKEN INTO CONSIDERATION")
        recommendations = recommender.recommend_for_user(user_embedding, top_n=1000)
        # output_path = f"/tmp/recommendations_{user_id}.csv"
        # recommendations.to_csv(output_path, index=False)
        # print(f"Recommendations written to: {output_path}")
        # Convert DataFrame to list of dictionaries
        recommendations_list = recommendations.to_dict(orient="records")

        # print(f"RECOMMENDATIONS:{recommendations_list}")
        return {
            'jobs' : recommendations_list
        }
    except Exception as e:
        import traceback
        print("bad ERROR OCCURRED:")
        traceback.print_exc() 
        return {"error": str(e)}




# Define request schema
class ExampleRequestGen(BaseModel):
    blurb: str
generator = llmRecommender(supabase)
@app.post("/generate/")
def generate(req: ExampleRequestGen):
    try:
        # Load recommender once

        print("JUST STARTING")
        blurb = req.blurb
        print(f"USERID:{blurb}")
        
        print("INIT IS DONE")
        generator.fetch_data()
        print("Before MODEL FIT")
        generator.load_data()
        print("MODEL FIT IS DONE")
        generator.build_qa_chain()
        print("USER RECOMMENDATIONS TAKEN INTO CONSIDERATION")
        gen_recommendations = generator.recommend(blurb)
        # output_path = f"/tmp/recommendations_{user_id}.csv"
        # recommendations.to_csv(output_path, index=False)
        # print(f"Recommendations written to: {output_path}")
        # Convert DataFrame to list of dictionaries
        recommendations_list = gen_recommendations.to_dict(orient="records")

        # print(f"RECOMMENDATIONS:{recommendations_list}")
        return {
            'jobs' : recommendations_list
        }
    except Exception as e:
        import traceback
        print("bad ERROR OCCURRED:")
        traceback.print_exc() 
        return {"error": str(e)}
