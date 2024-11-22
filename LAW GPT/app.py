from flask import Flask, render_template, request, jsonify
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Load the model for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for Q&A

# Load and cache the questions and answers from JSON
def load_legal_context():
    with open("ipc_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Extract questions and answers separately
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers

# Load questions and answers and compute embeddings at startup
questions, answers = load_legal_context()
question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)

# Function to find the best answer based on semantic similarity with debugging
def find_best_answer(user_question):
    # Encode the user's question
    user_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    
    # Compute similarity scores
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    
    # Get the index and value of the highest similarity score
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()
    
    # Debugging prints
    print(f"User question: {user_question}")
    print(f"Best match question: {questions[best_match_idx]}")
    print(f"Best match score: {best_match_score}")
    
    # Increase threshold to ensure relevance (e.g., 0.7)
    if best_match_score > 0.7:
        return answers[best_match_idx]
    else:
        return "I'm sorry, I couldn't find an answer to that question."

# Route for chatbot UI
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chatbot queries
@app.route('/get-response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')
    
    if user_input:
        response = find_best_answer(user_input)
        return jsonify({'response': response})
    
    return jsonify({'response': "Sorry, I didn't understand that. Can you clarify your question?"})

if __name__ == '__main__':
    app.run(debug=True)
