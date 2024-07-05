import streamlit as st

# Set page configuration
st.set_page_config(page_icon="assets\\favicon.ico", page_title='DeepMLeet')

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
    "Deep Learning [In Progress]": DeepLearning
}

# Sidebar for navigation for navigating 
url = 'https://www.deep-ml.com/'
st.warning("check my new site [deep-ml](%s)" % url, icon="⚠️")

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Sections", list(pages.keys()))

# Page selection
page = pages[selection]
page.app()
