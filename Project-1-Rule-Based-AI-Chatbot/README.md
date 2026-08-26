# Rule-Based AI Chatbot

## Objective

A simple rule-based chatbot built with Python that responds to predefined user inputs using a dictionary-based knowledge base.

## Technologies Used

* Python 3

## How It Works

1. The chatbot runs in a continuous loop and waits for user input.
2. The user's input is converted to lowercase and stripped of extra whitespace.
3. The cleaned input is matched against predefined responses stored in a dictionary.
4. The dictionary `.get()` method is used to retrieve the response.
5. If no matching input is found, the chatbot provides a fallback response.
6. The chatbot exits when the user enters `bye`, `exit`, or `quit`.

## Features

* Handles greetings such as `hello`, `hi`, and `hey`
* Responds to common questions
* Handles `help` and thank-you messages
* Uses a dictionary-based knowledge base
* Uses `.get()` for response lookup
* Provides a fallback response for unknown inputs
* Runs continuously until an exit command is entered
* Requires no external libraries

## How to Run

Open a terminal in the project folder and run:

```bash
python chatbot.py
```

## Example Conversation

```text
========================================
  Rule-Based AI Chatbot
  Type 'bye' to exit
========================================

You: Hello
Bot: Hey there! How can I help you?

You: What is your name
Bot: I'm ChatBot, your virtual assistant.

You: help
Bot: Sure! Try asking me something like 'how are you' or 'what is your name'.

You: how are you
Bot: I'm doing great, thanks for asking!

You: thanks
Bot: Happy to help!

You: bye
Bot: Goodbye! Have a great day!
```

## Project Purpose

This project demonstrates basic AI concepts through explicit rule-based decision-making, including control flow, input handling, dictionary lookup, and continuous interaction.
