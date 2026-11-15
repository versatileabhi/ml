def chatbot():
    print("Chatbot Started")
    print("Type 'bye' to exit")
    
    while True:
        user = input("You: ").lower()
        
        if user == "hello":
            print("Bot: Hello! How can I help you?")
        elif user == "how are you":
            print("Bot: I am fine.")
        elif user == "your name":
            print("Bot: I am an AI Chatbot.")
        elif user == "bye":
            print("Bot: Goodbye!")
            break
        else:
            print("Bot: Sorry, I don't understand.")

# Run the chatbot function
if __name__ == "__main__":
    chatbot()