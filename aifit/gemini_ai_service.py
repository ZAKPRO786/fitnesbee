# services/gemini_ai_service.py
import os
from decouple import config
import google.generativeai as genai

# Load the API key from environment variables
API_KEY = config('GENERATIVE_AI_KEY', default=None)

if API_KEY is None:
    raise ValueError('GENERATIVE_AI_KEY environment variable not set')

# Configure the generative AI SDK
genai.configure(api_key=API_KEY)

# Default generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the generative model with a system instruction
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="""You are an expert mental therapist, specializing in emotional support and psychological guidance. You should only respond to emotional, mental, and skill-related prompts; otherwise, tell users you are here for mental support. Your goal is to help users navigate through their mental and emotional challenges with compassion, understanding, and expert advice..."""
)


def chat_with_gemini(user_input, chat_history):
    # Create or continue a chat session
    chat_session = model.start_chat(history=chat_history)

    # Send user input and get a response from the model
    response = chat_session.send_message(user_input)

    return response.text, chat_session.history

