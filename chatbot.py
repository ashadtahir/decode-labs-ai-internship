# Rule-Based AI Chatbot
# A simple chatbot that uses a dictionary to map user inputs to responses.

# Knowledge base: maps keywords/phrases to bot responses
responses = {
    "hello": "Hey there! How can I help you?",
    "hi": "Hello! What's on your mind?",
    "hey": "Hey! Nice to chat with you.",
    "how are you": "I'm doing great, thanks for asking!",
    "what is your name": "I'm ChatBot, your virtual assistant.",
    "what can you do": "I can answer simple questions and keep you company!",
    "help": "Sure! Try asking me something like 'how are you' or 'what is your name'.",
    "thank you": "You're welcome!",
    "thanks": "Happy to help!",
    "bye": "Goodbye! Have a great day!",
    "exit": "See you later!",
    "quit": "Bye! Take care!",
}

# Fallback response when input doesn't match anything
FALLBACK = "Sorry, I don't understand that yet."

def get_response(user_input):
    # Convert input to lowercase and remove extra spaces
    cleaned = user_input.lower().strip()
    # Look up the response using .get() with a fallback
    return responses.get(cleaned, FALLBACK)

def main():
    print("=" * 40)
    print("  Rule-Based AI Chatbot")
    print("  Type 'bye' to exit")
    print("=" * 40)

    # Continuous chat loop
    while True:
        user_input = input("\nYou: ").strip()

        # Skip empty inputs
        if not user_input:
            print("Bot: Please say something!")
            continue

        response = get_response(user_input)
        print(f"Bot: {response}")

        # Stop the loop on exit commands
        if user_input.lower() in ["bye", "exit", "quit"]:
            break

if __name__ == "__main__":
    main()
