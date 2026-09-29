from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    host=os.getenv("PINECONE_HOST")
)

if __name__ == '__main__':
    print("Ingesting...")