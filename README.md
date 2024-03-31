# DeepMLeet

Welcome to **DeepMLeet**: a cutting-edge platform tailored for Machine Learning enthusiasts and practitioners. Drawing inspiration from the challenge-based approach of platforms like LeetCode, DeepMLeet brings forth an interactive environment specifically crafted for Deep Learning and Machine Learning challenges.

## Overview

DeepMLeet leverages the convenience and power of Streamlit, coupled with the code execution capabilities of pistonpy, to offer users a seamless experience in solving, testing, and submitting solutions for a variety of ML challenges. From data manipulation to complex algorithm implementations, our platform supports a broad spectrum of ML domains.

### Features

- **Interactive Code Editor:** Utilize the streamlit-ace editor for an enhanced coding experience, complete with syntax highlighting and theme customization.
- **Real-Time Execution and Feedback:** Submit your code to be executed by pistonpy, offering immediate feedback and results directly within the platform.
- **Dynamic Problem Selection:** Choose from a growing list of ML problems, each with a detailed description, starter code, and test cases.
- **Instant Test Case Evaluation:** Run your solutions against predefined test cases to ensure accuracy and performance, with immediate pass/fail feedback.
- **Solution Insights:** Access the solution for learning purposes after attempting the problems, fostering understanding and improvement.

## Running the DeepMLeet Streamlit App

Follow this guide to set up your environment and run the Streamlit app from the DeepMLeet repository.

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### Setup

#### 1. Clone the Repository

Clone the DeepMLeet repository to your local machine using the terminal or command prompt:

```bash
git clone https://github.com/moe18/DeepMLeet.git
cd DeepMLeet
```
#### 2.Create a Conda Environment
Create a new Conda environment using Python 3.12.2. Replace deepmleetenv with your preferred environment name if desired:
```bash
conda create --name deepmleetenv python=3.12.2
conda activate deepmleetenv
```

#### 3. Install Dependencies
With the Conda environment activated, install the required dependencies, including Streamlit. This assumes there is a requirements.txt file present in the repository:

``` bash
pip install -r requirements.txt
```

### Running the App
With the environment set and dependencies installed, run the Streamlit app by executing:
```bash
streamlit run app/main.py
```
This command assumes that main.py is located within an app directory. Adjust the path accordingly if your file structure is different.


## How to Contribute

We are excited to have you contribute to the DeepMLeet project! Whether you're looking to improve the UI, add new problems, or enhance existing ones, your contributions are welcome.

For detailed instructions on how to contribute, including guidelines and best practices, please see our [contribute.md](contribute.md) page. This guide provides all the information you need to get started, including how to fork the repository, make changes, and submit your contributions for review.

Your insights, improvements, and additions are what make this project thrive. We look forward to seeing your contributions!


