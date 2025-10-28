import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Get API key, model URL, and model name from environment variables
api_key = os.getenv("OPENAI_API_KEY")
model_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL")

# Initialize the OpenAI client with the custom base URL
client = OpenAI(
    api_key=api_key,
    base_url=model_url
)

# Define your personas
PERSONAS = {
    "python_tutor": {
        "name": "Python Tutor",
        "system_message": """You are a patient and encouraging Python programming tutor.
Your goal is to help students learn Python through clear explanations and examples.
Guidelines:
- Always provide code examples to illustrate concepts
- Break down complex topics into simple steps
- Encourage questions and experimentation
- Use analogies to explain difficult concepts
- Be supportive and positive"""
    },
    "shakespeare": {
        "name": "Shakespearean Translator",
        "system_message": """You are a Shakespearean translator. Convert modern English into the style of William Shakespeare.
Guidelines:
- Use thee, thou, thy, thine appropriately
- Employ archaic verb forms (doth, hath, art)
- Include poetic flourishes and metaphors
- Maintain the meaning of the original message
- Keep responses relatively brief and elegant"""
    },
    "socratic_teacher": {
        "name": "Socratic Teacher",
        "system_message": """You are a Socratic teacher. You answer questions with questions to encourage critical thinking.
Guidelines:
- Never give a direct answer.
- Always respond with a question that guides the user to their own conclusion.
- Your questions should be thought-provoking and related to the original query."""
    },
    "eli5_explainer": {
        "name": "ELI5 Explainer",
        "system_message": """You explain complex topics as if to a 5-year-old.
Guidelines:
- Use simple words and short sentences.
- Use analogies that a child can understand.
- Avoid jargon and technical terms."""
    },
    "technical_writer": {
        "name": "Technical Writer",
        "system_message": """You are a technical writer. You provide concise, bullet-pointed technical explanations.
Guidelines:
- Be clear, concise, and to the point.
- Use bullet points or numbered lists to structure information.
- Focus on facts and technical details."""
    }
}

class PersonaChat:
    """Manages conversation with a specific persona."""
    def __init__(self, persona_key):
        self.persona = PERSONAS[persona_key]
        self.history = [
            {"role": "system", "content": self.persona["system_message"]}
        ]

    def send_message(self, user_message, stream=True):
        """Send a message and get streaming response."""
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=model_name,
            messages=self.history,
            max_tokens=200,
            stream=stream
        )
        
        full_response = ""
        if stream:
            print(f"\n{self.persona['name']}: ", end="", flush=True)
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()  # New line after response
        else:
            full_response = response.choices[0].message.content
            print(f"\n{self.persona['name']}: {full_response}")


        self.history.append({"role": "assistant", "content": full_response})
        return full_response

    def reset(self):
        """Reset conversation history."""
        self.history = [
            {"role": "system", "content": self.persona["system_message"]}
        ]
    
    def export_history(self, filename="conversation_history.json"):
        """Export conversation history to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"Conversation history exported to {filename}")

    def token_count(self):
        """Returns the total number of tokens in the conversation history."""
        return sum(len(message["content"]) for message in self.history)


def main():
    """
    Main function to run the multi-persona chat system.
    """
    print("\n" + "="*50)
    print("MULTI-PERSONA CHAT SYSTEM")
    print("="*50)

    persona_keys = list(PERSONAS.keys())
    
    def print_personas():
        print("Available Personas:")
        for i, key in enumerate(persona_keys, 1):
            print(f"{i}. {PERSONAS[key]['name']}")

    print_personas()

    current_persona_key = persona_keys[0]
    chat = PersonaChat(current_persona_key)

    while True:
        print("\n" + "="*50)
        print(f"Current Persona: {chat.persona['name']}")
        print("Commands:")
        print("- /switch <number> - Switch to a different persona")
        print("- /compare <persona1_num> <persona2_num> ... - Compare responses")
        print("- /reset - Reset current conversation")
        print("- /export - Export conversation history")
        print("- /quit - Exit")
        print("="*50)

        user_input = input("You: ")

        if user_input.lower() == "/quit":
            break
        elif user_input.lower().startswith("/switch"):
            try:
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit():
                    new_persona_index = int(parts[1]) - 1
                    if 0 <= new_persona_index < len(persona_keys):
                        current_persona_key = persona_keys[new_persona_index]
                        chat = PersonaChat(current_persona_key)
                        print(f"Switched to: {chat.persona['name']}")
                    else:
                        print("Invalid persona number.")
                else:
                    print("Invalid command format. Use /switch <number>")
            except (ValueError, IndexError):
                print("Invalid command format.")
        elif user_input.lower().startswith("/compare"):
            try:
                parts = user_input.split()
                if len(parts) > 2:
                    persona_indices = [int(p) - 1 for p in parts[1:]]
                    question = input("Enter the question to ask all personas: ")
                    
                    for index in persona_indices:
                        if 0 <= index < len(persona_keys):
                            persona_key = persona_keys[index]
                            temp_chat = PersonaChat(persona_key)
                            temp_chat.send_message(question, stream=False)
                        else:
                            print(f"Invalid persona number: {index + 1}")
                else:
                    print("Invalid command format. Use /compare <persona1_num> <persona2_num> ...")
            except (ValueError, IndexError):
                print("Invalid command format.")

        elif user_input.lower() == "/reset":
            chat.reset()
            print("Conversation reset.")
        elif user_input.lower() == "/export":
            chat.export_history(f"{current_persona_key}_history.json")
        else:
            chat.send_message(user_input)


if __name__ == "__main__":
    main()
