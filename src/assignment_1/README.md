# Assignment 1
### Chat with AI
you can select the wanted LLM from OptoGpt and start chatting!, you can also adjust the temprature.

### Temperature Experimentation Dashboard
the Temprature Experimentation Dashboard its goal to test the same prompt with different temperature values. you can select pre-fixed categories or add the custom one.


### Multi-Persona Chat System
its goal to answer the prompt with different personas. you can switch, compare, reset and more.

### Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (on Unix or MacOS) or `.\venv\Scripts\activate` (on Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file from the `.env.example` and add your OpenAI API key.

### How to Run Each Program
- **Temperature Experimentation Dashboard**: `python src/temperature_dashboard.py`
- **Multi-Persona Chat System**: `python src/persona_chat.py`
- **Chat with AI**: `python src/app.py`

### Dependencies
- openai>=1.0.0
- python-dotenv>=1.0.0
- matplotlib>=3.0.0





