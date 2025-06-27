### Hallucination Detector
This repository contains the implementation of a Hallucination Detector, a Streamlit-based application designed to evaluate the reliability of language model outputs by detecting potential hallucinations (inaccurate or fabricated responses). The tool leverages the uqlm library and integrates with OpenRouter models to provide a user-friendly interface for quantifying uncertainty in AI-generated text, ensuring trustworthy results for critical applications.
Overview
The Hallucination Detector assesses the reliability of language model outputs using multiple detection methods. It is built to support various OpenRouter models (e.g., DeepSeek, Gemini, LLaMA) and provides confidence scores to indicate the likelihood of hallucinations, making it suitable for high-stakes scenarios such as legal, medical, or financial applications.
## Key Features

Model Flexibility: Supports multiple OpenRouter models for diverse use cases.
Detection Methods:
Black-Box: Compares multiple responses for consistency.
White-Box: Analyzes token probabilities for confidence.
LLM-as-a-Judge: Uses secondary LLMs to evaluate outputs.
Ensemble: Combines methods for robust evaluation.


User Interface: Configurable via a Streamlit sidebar for selecting model, temperature, and scoring method; supports custom or predefined prompts.
Visualization: Displays confidence scores (0–1) in a bar chart, where:

0.8: Likely factual.


0.5–0.8: Possible inaccuracies.
≤0.5: Likely hallucinations.


Performance: Asynchronous processing for efficient execution, with Ensemble Scorer being computationally intensive.

## Value

Trustworthy AI: Ensures reliable outputs for decision-making in critical applications.
Risk Mitigation: Identifies and flags potential inaccuracies, reducing errors in AI-driven processes.
User Accessibility: Intuitive interface requiring minimal technical expertise.
Scalable Insights: Configurable settings for tailoring to specific industries or applications.

## Prerequisites
# Requirements

OpenRouter API Key: Required for model access. Obtain one from OpenRouter.
Internet Connection: Necessary for API calls.
Python Version: Python 3.8 or higher.

# Dependencies
Install the required Python packages:
pip install streamlit langchain uqlm

# Key Libraries

streamlit: For building the interactive user interface.
langchain: For integration with OpenRouter models.
uqlm: For hallucination detection and confidence scoring.

Setup and Installation

Clone this repository:
git clone https://github.com/Avinashhmavi/Quantization-using-different-methods/new/main/scripts/FineTuning/Hallucination-Detector
cd hallucination-detector


Install dependencies:
pip install -r requirements.txt


Set up the OpenRouter API key:

Create a .env file in the project root and add:OPENROUTER_API_KEY=your-api-key-here


Alternatively, set the API key as an environment variable:export OPENROUTER_API_KEY=your-api-key-here




Run the Streamlit application:
streamlit run app.py



## Usage

Launch the Application:
After running the Streamlit command, the application will open in your default web browser.


## Configure Settings:
Use the sidebar to select the OpenRouter model, temperature, and scoring method (Black-Box, White-Box, LLM-as-a-Judge, or Ensemble).
Enter a custom prompt or select a predefined one.


## Evaluate Outputs:
Submit a prompt to generate a response from the selected model.
The application will display the LLM response, confidence scores, and a bar chart visualizing the hallucination risk.


## Interpret Results:
Confidence scores >0.8 indicate likely factual responses.
Scores between 0.5–0.8 suggest possible inaccuracies.
Scores ≤0.5 indicate likely hallucinations.



## Example
# Example usage within app.py
import streamlit as st
from langchain.llms import OpenRouter
from uqlm import HallucinationDetector

# Initialize OpenRouter model
llm = OpenRouter(model="deepseek", api_key=st.secrets["OPENROUTER_API_KEY"])

# Initialize Hallucination Detector
detector = HallucinationDetector(method="ensemble")

# Get user input
prompt = st.text_input("Enter your prompt:")
if prompt:
    response = llm(prompt)
    score, details = detector.evaluate(response)
    st.write(f"Response: {response}")
    st.write(f"Confidence Score: {score}")
    st.bar_chart({"Confidence": [score]})

## Notes

The Ensemble Scorer is computationally intensive and may require more resources for large-scale use.
Ensure a stable internet connection for seamless API calls to OpenRouter.
The uqlm library is critical for hallucination detection; ensure it is correctly installed and compatible with your Python environment.

## Future Improvements

Additional Models: Expand support for more language models beyond OpenRouter.
Optimized Performance: Reduce computational overhead for the Ensemble Scorer.
Enhanced Visualization: Add more detailed visualizations for confidence score breakdowns.
Custom Scoring Methods: Allow users to define custom hallucination detection criteria.

## Conclusion
The Hallucination Detector provides a robust tool for ensuring the reliability of AI-generated text, making it a valuable asset for decision-makers in high-stakes applications. Its intuitive interface and flexible detection methods enable broad adoption across industries.
License
This project is licensed under the MIT License. See the LICENSE file for details.
