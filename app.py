import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Pre-trained model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a Q&A pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Predefined context about OOP in C++ to help the chatbot answer user questions
oop_context = """
Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, 
which contain data and methods. Key concepts in OOP include encapsulation, inheritance, polymorphism, 
and abstraction. In C++, classes and objects are used to implement these concepts. For example:
1. Encapsulation: Data hiding using private and public access modifiers.
2. Inheritance: Creating derived classes from base classes to reuse code.
3. Polymorphism: Using function overloading or virtual functions to achieve dynamic behavior.
4. Abstraction: Hiding complex implementation details and exposing only essential features.
"""

# Function to generate a response using the pre-trained model and context
def generate_response(user_query):
    try:
        # Use the pipeline to answer the question based on the context
        response = qa_pipeline({
            'question': user_query,
            'context': oop_context
        })
        return response['answer']
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI: Set up the page title and a brief description
st.title("OOP in C++ Chatbot")
st.write("""
    This chatbot answers questions about Object-Oriented Programming (OOP) concepts in C++.
    You can ask about Encapsulation, Inheritance, Polymorphism, Abstraction, and other OOP-related topics.
    Type 'exit' to end the conversation.
""")

# Function to interact with the chatbot
def chatbot():
    user_input = st.text_input("Ask your OOP question: ")
    
    if user_input:
        if user_input.lower() == 'exit':
            st.write("Goodbye!")
        else:
            # Generate the chatbot's response based on user input
            answer = generate_response(user_input)
            st.write(f"**Chatbot Answer:** {answer}")

# Start the chatbot
chatbot()

