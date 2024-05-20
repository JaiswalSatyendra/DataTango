from fastapi import FastAPI
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# MongoDB connection
mongo_uri = os.getenv("MONGODB_URL")
client = MongoClient(mongo_uri)
db = client["convertml-test"]
collection = db["datatango"]

@app.get("/questions/{user_id}/{survey_id}")
def get_questions(user_id: str, survey_id: str):
    # Find the document containing the questions
    document = collection.find_one({"user_id": user_id, "survey_id": survey_id})

    if document:
        # Extract the questions from the document
        questions = document["questions"]
        return {"questions": questions}
    else:
        return {"message": "Document not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
