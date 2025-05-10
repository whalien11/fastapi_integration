import pandas as pd
import numpy as np
import openai
import tiktoken
import tempfile
import os
import openai

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


class llmRecommender:
    def __init__(self, supabase):
        self.supabase = supabase
        self.df = None
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.jobs_table = 'jobs'

    def fetch_data(self):
        # fetch opportunities
        response = self.supabase.table(self.jobs_table).select(
            'id, title, organization, description, date, location, skills, requirement'
        ).execute()
        self.df = pd.DataFrame(response.data)
        self.preprocess()

    def preprocess(self):
        self.df['summarized'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['location'].fillna('') + ' ' +
            self.df['skills'].fillna('') + ' ' +
            self.df['requirement'].fillna('')
    )

    def load_data(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmp_file:
            self.df[['summarized']].to_csv(tmp_file.name, index=False)
            tmp_file.flush()
            print("csv data okay:", tmp_file.name)

            loader = CSVLoader(file_path=tmp_file.name)
            documents = loader.load()
        # # Convert DataFrame to CSV format
        # csv_data = self.df['summarized'].to_csv(index=False)
        # print("csv data okay")
        # # Create a CSVLoader instance
        # loader = CSVLoader(file_path=csv_data)
        # # Load documents from the CSV data
        # documents = loader.load()
        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        # Create a Chroma vector store from the texts
        self.vectorstore = Chroma.from_documents(texts, self.embeddings)

    def build_qa_chain(self):
        # Create a retriever from the vector store
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        # Create a RetrievalQA chain
        self.qa_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.retriever
        )

    def recommend(self, blurb):
        # Build the QA chain
        self.build_qa_chain()

        # Get recommendations based on the query
        # the query is the blurb plus a given prompt
        self.query = f"Based on the following blurb: {blurb}, Give 20 suggestions. (only the title)"
        response = self.qa_chain.invoke({"query": self.query})
        text = response['result']
        print(text)
        # parse the result to extract the recommended opportunities
        recommended_opportunities = pd.DataFrame()
        for line in text.split('\n'):
            # Check if the line starts with a number followed by a period and space
            if line.strip() and line[0].isdigit() and '. ' in line:
                # Extract the content after the number and period
                title = line.split('. ', 1)[1]
                
                try:
                    # Find the corresponding row in the DataFrame
                    row = self.df[self.df['title'].str.contains(title)]
                    if not row.empty:
                        recommended_opportunities = pd.concat([recommended_opportunities, row])
                except Exception as e:
                    print(f"Error finding row for title '{title}': {e}")
                    continue

        # final output: list of opportunities with every info on it
        return recommended_opportunities
                        
