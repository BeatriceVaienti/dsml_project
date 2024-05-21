import streamlit as st
import openai
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification

# Set your OpenAI API key
openai.api_key = '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'

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('C:\\Users\\berret_c\\Downloads\\best_model\\best_model\\camembert_full_515')
model = CamembertForSequenceClassification.from_pretrained('C:\\Users\\berret_c\\Downloads\\best_model\\best_model\\camembert_full_515')

# Initialize session state
if "level" not in st.session_state:
    st.session_state.level = 0
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "test_finished" not in st.session_state:
    st.session_state.test_finished = False

levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
level_values = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}

# Generate a new question for the current level if needed
def generate_question(level):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a French question for level {levels[level]}."}
        ]
    )
    return response['choices'][0]['message']['content']

# Debugging helper
def log_state():
    st.write(f"Current level: {st.session_state.level}")
    st.write(f"User answer: {st.session_state.user_answer}")
    st.write(f"Question: {st.session_state.question}")
    st.write(f"Test finished: {st.session_state.test_finished}")

# Process user answer and level advancement
def process_answer():
    # Tokenize the user's answer
    inputs = tokenizer(st.session_state.user_answer, return_tensors='pt')

    # Make a prediction with your model
    outputs = model(**inputs)

    # Extract the predicted level from the outputs
    predicted_level = outputs.logits.argmax(-1).item()

    st.write(f"Predicted level: {levels[predicted_level]}")  # Debugging statement

    if predicted_level < st.session_state.level:
        st.write(f'Your French level is {levels[st.session_state.level]}.')
        st.session_state.test_finished = True
    else:
        st.session_state.level += 1
        st.session_state.question = generate_question(st.session_state.level)  # Generate a new question here

# Generate the initial question if not already set
if not st.session_state.question:
    st.session_state.question = generate_question(st.session_state.level)

# Display the question and get the user's answer
if not st.session_state.test_finished:

    st.write(f'Question for level {levels[st.session_state.level]}: {st.session_state.question}')

    # Display the text input widget with a unique key for each level
    st.session_state.user_answer = st.text_input('Your answer', key=f'user_answer_{st.session_state.level}')

    # Check if an answer has been entered
    if st.session_state.get(f'user_answer_{st.session_state.level}'):
        process_answer()
        st.experimental_rerun()  # Add this line to rerun the script
else:
    st.write(f'Test finished. Your final French level is {levels[st.session_state.level]}.')

# Log the state for debugging
log_state()