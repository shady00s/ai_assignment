from flask import Flask, render_template, request
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get API key, model URL, and model name from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("OPENAI_BASE_URL")
base_model = os.getenv("OPENAI_MODEL")

# Initialize the OpenAI client with the custom base URL
client = OpenAI(
    api_key=api_key,
    base_url=model_url
)

# Store prompt history
prompt_history = []

def get_available_models():
    """Fetches the available models from the LiteLLM API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{model_url}/models", headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        models_data = response.json()
        # The model names are in the 'id' field of each object in the 'data' list
        model_ids = [model['id'] for model in models_data['data']]
        return model_ids
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        # Fallback to a default list if the API call fails
        return [base_model]


@app.route('/', methods=['GET', 'POST'])
def index():
    models = get_available_models()
    response_text = ""
    
    if request.method == 'POST':
        prompt = request.form['prompt']
        model = request.form['model']
        temperature = float(request.form['temperature'])
        
        # Add prompt to history
        if prompt and prompt not in prompt_history:
            prompt_history.append(prompt)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=150
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            response_text = f"An error occurred: {e}"

    return render_template('index.html', 
                         models=models, 
                         response=response_text, 
                         default_model=base_model,
                         prompts=prompt_history)

if __name__ == '__main__':
    app.run(debug=True)