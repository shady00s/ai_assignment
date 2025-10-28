import os
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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

def test_temperature(prompt, temperatures=[0.0, 0.5, 0.7, 1.0, 1.5]):
    """
    Test the same prompt with different temperature values.
    Args:
        prompt: The user prompt to test
        temperatures: List of temperature values to test
    Returns:
        Dictionary mapping temperature to response
    """
    results = {}
    for temp in temperatures:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=150  # Prevent rambling
            )
            results[temp] = response.choices[0].message.content
        except Exception as e:
            results[temp] = f"Error: {e}"
    return results

def analyze_results(prompt, results):
    """
    Analyzes the results from the temperature test.
    """
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)

    lengths = {temp: len(res) for temp, res in results.items()}
    
    # Consistency
    consistency_temps = []
    base_response = results[0.0]
    for temp, response in results.items():
        if response == base_response:
            consistency_temps.append(str(temp))
    
    if len(consistency_temps) > 1:
        print(f"- Consistency: Temps {', '.join(consistency_temps)} produced identical results.")
    else:
        print("- Consistency: All temperatures produced different results.")

    # Creativity/Diversity
    max_len_temp = max(lengths, key=lengths.get)
    print(f"- Creativity: Temp {max_len_temp} produced the longest and potentially most creative response.")

    # Recommendations
    print("- Recommendation:")
    if "what is" in prompt.lower() or "who is" in prompt.lower():
        print("  - Use temperature 0.0-0.5 for factual queries to get reliable answers.")
    elif "write a" in prompt.lower() or "tell me a story" in prompt.lower():
        print("  - Use temperature 0.7-1.0 for creative writing to get more imaginative results.")
    elif "function" in prompt.lower() or "code" in prompt.lower():
        print("  - Use temperature 0.2-0.5 for code generation to get working and predictable code.")
    elif "if" in prompt.lower() and "then" in prompt.lower():
         print("  - Use temperature 0.3-0.7 for reasoning tasks.")
    else:
        print("  - Use lower temperatures for predictability and higher temperatures for creativity.")


def plot_results(prompt, results):
    """
    Plots the results of the temperature experiment.
    """
    temperatures = list(results.keys())
    response_lengths = [len(res) for res in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(temperatures)), response_lengths, tick_label=[str(t) for t in temperatures])
    plt.xlabel("Temperature")
    plt.ylabel("Response Length (characters)")
    plt.title(f"Temperature vs. Response Length for:\n'{prompt}'")
    plt.show()


def main():
    """
    Main function to run the temperature experimentation dashboard.
    """
    test_prompts = {
        "Factual": "What is the capital of France?",
        "Creative": "Write a short story about a robot who discovers music.",
        "Code": "Write a Python function to calculate the factorial of a number.",
        "Reasoning": "If all humans are mortal, and Socrates is a human, what can we conclude about Socrates?",
        "Custom": ""
    }

    while True:
        print("\n" + "="*50)
        print("TEMPERATURE EXPERIMENTATION DASHBOARD")
        print("="*50)
        print("Select a prompt category:")
        for i, category in enumerate(test_prompts.keys(), 1):
            print(f"{i}. {category}")
        
        choice = input("Enter your choice (1-5): ")

        if not choice.isdigit() or not 1 <= int(choice) <= 5:
            print("Invalid choice. Please try again.")
            continue

        category = list(test_prompts.keys())[int(choice) - 1]
        
        if category == "Custom":
            prompt = input("Enter your custom prompt: ")
        else:
            prompt = test_prompts[category]

        print(f"\nPrompt: \"{prompt}\"")
        print(f"Category: {category}")
        print("-" * 50)

        results = test_temperature(prompt)

        for temp, response in results.items():
            print(f"Temperature {temp}:")
            print(f"\"{response}\"")
            print(f"(Length: {len(response)} characters)")
            print("-" * 20)

        analyze_results(prompt, results)
        plot_results(prompt, results)

        another = input("\nRun another experiment? (y/n): ")
        if another.lower() != 'y':
            break

if __name__ == "__main__":
    main()
