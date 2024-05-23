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

# Define the image paths
image_path_left = 'C:\\Users\\berret_c\\Documents\\GitHub\\dsml_project\\Berre.png'
image_path_right = 'C:\\Users\\berret_c\\Documents\\GitHub\\dsml_project\\Vaient.png'

# Layout for images and title
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<p style='text-align: center;'>Berre</p>", unsafe_allow_html=True)
    st.image(image_path_left, width=150)  # Adjust width as needed

with col2:
    st.markdown("<h1 style='text-align: center;'>Welcome to Berrevaient's French Proficiency Test!</h1>", unsafe_allow_html=True)

with col3:
    st.markdown("<p style='text-align: center;'>Vaient</p>", unsafe_allow_html=True)
    st.image(image_path_right, width=150)  # Adjust width as needed

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

    # Check if answer predicts a level at or below the current level, or one level above
    if predicted_level <= st.session_state.level + 1:
        st.session_state.level = min(st.session_state.level + 1, len(levels) - 1)  # Increment level but do not exceed max
    else:
        st.session_state.level = min(predicted_level, len(levels) - 1)  # Jump to predicted level but do not exceed max

    # Check if maximum level is reached or exceeded
    if st.session_state.level >= len(levels) - 1:
        st.session_state.test_finished = True
        st.write(f'Your final French level is {levels[-1]}.')  # Provide final level
    else:
        st.session_state.question = generate_question(st.session_state.level)  # Generate a new question if not at max level

# Generate the initial question if not already set
if not st.session_state.question and not st.session_state.test_finished:
    st.session_state.question = generate_question(st.session_state.level)

# Display the question and get the user's answer
if not st.session_state.test_finished:
    
    st.markdown("""
    **Answer the following questions in French to determine your language proficiency level.**  
    **This will help you to find your Tandem Match and tailor your learning program based on your conversations.**
    """)

    st.write(f'Question for level {levels[st.session_state.level]}: {st.session_state.question}')

    # Display the text input widget with a unique key for each level
    st.session_state.user_answer = st.text_input('Your answer', key=f'user_answer_{st.session_state.level}')

    # Display the button for submitting the answer
    if st.button('Submit'):
        process_answer()
        st.experimental_rerun()  # Add this line to rerun the script

    # Display the slider as a progress bar
    st.slider('Your Current Level:', 0, len(levels) - 1, st.session_state.level, format=levels[st.session_state.level])

else:
    st.write(f'Test finished. Your final French level is {levels[st.session_state.level]}.')

# Log the state for debugging
log_state()