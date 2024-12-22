from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS  # Import CORS from flask_cors
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (Allows cross-origin requests from frontend)
CORS(app)

# Set up logging for better error tracking
logging.basicConfig(level=logging.DEBUG)

# Initialize the Hugging Face question-answering pipeline with a specific model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define FAQ/context with information about clinic hours
faq_context = """
1. Clinic Hours:
   - Our dental clinic is open from 10 AM to 5 PM every day, Monday through Friday.
   - We are available from 10:00 AM to 5:00 PM, Monday to Friday.
   - You can visit us anytime between 10 AM and 5 PM for your dental care needs.
   - Our working hours are from 10 AM to 5 PM, Monday to Friday.
   - What time do we open? From 10 AM to 5 PM every weekday.
   - The clinic opens at 10 AM and closes at 5 PM, Monday through Friday.
   - We’re here for you Monday to Friday, from 10 AM to 5 PM, for all dental services.
   - Visit us anytime between 10 AM and 5 PM, Monday through Friday.
   - Our doors are open from 10 AM to 5 PM, Monday through Friday, for all your dental needs.
   - Whether it’s a routine check-up or a dental emergency, we’re open from 10 AM to 5 PM on weekdays.
   - We are closed on weekends and public holidays, but we’re happy to assist you during our regular hours.
   - Our weekend hours are limited, so please schedule any non-urgent appointments for weekdays.
   - If you need an appointment outside of our standard hours, please inquire about our extended hours or availability.
   - Evening or weekend appointments may be available for urgent dental care on a case-by-case basis.
"""

# Define the /chatbot endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Get the user's question from the incoming request
    user_question = request.json.get("question")
    
    # If no question is provided, return an error
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Perform question answering using Hugging Face pipeline
        result = qa_pipeline(question=user_question, context=faq_context)
        
        # Only return an answer if the confidence score is above 0.2
        if result['score'] > 0.2:
            return jsonify({"answer": result['answer']})
        else:
            return jsonify({"answer": "Please get help about that topic at support@example.com."})
    
    except Exception as e:
        # Log the error and return a friendly message to the user
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

# Make sure the app runs in local development if the script is run directly
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
