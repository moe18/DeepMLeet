import streamlit as st

# Import your page modules
import LinAlg
import MachineLearning
import DeepLearning
import about

# Define pages in the app
pages = {
    "About": about,
    "Linear Algebra": LinAlg,
    "Machine Learning": MachineLearning,
    "Deep Learning": DeepLearning
}

# Sidebar for navigation for navigating 
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Sections", list(pages.keys()))

# Page selection
page = pages[selection]
page.app()
