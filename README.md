# MaternAI-
 # Multilingual Maternity Chatbot

We developed and fine-tuned several models, including Mistral, Bio-Mistral, RAG (Retrieval-Augmented Generation), and Falcon, to create a chatbot tailored for maternity and pregnancy-related queries. After rigorous testing, we identified that Falcon outperformed other models in generating accurate, coherent, and contextually appropriate responses. This superior performance can be attributed to Falcon’s advanced architecture and its capability to effectively process domain-specific data. The chatbot leverages multilingual capabilities, enabling seamless communication in multiple Indian languages.

## Features
- Multilingual support for Indian languages including Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati, and Bengali.
- Integration of Google Translate for language translation and GTTS for text-to-speech conversion.
- Responsive, user-friendly web interface built with Tailwind CSS.
- Real-time voice input and output functionality for enhanced accessibility.
- Powered by Falcon’s fine-tuned maternity-specific language model.

## How to Run the Project
This guide will help you execute the project on Google Colab.

### Prerequisites
1. A Google account to access Google Colab.
2. An [ngrok](https://ngrok.com/) account for generating an authentication token.

### Steps to Execute

1. **Open Google Colab**
   - Upload the provided Python script or copy its contents into a new notebook.

2. **Install Required Dependencies**
   Run the following commands in Colab to install all necessary libraries and tools:
   ```python
   !pip install flask googletrans==3.1.0a0 gtts peft transformers torch accelerate
   !wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   !tar -xvf ngrok-v3-stable-linux-amd64.tgz
   !mv ngrok /usr/local/bin
   ```

3. **Obtain ngrok Authentication Token**
   - Sign up at [ngrok.com](https://ngrok.com/) if you haven’t already.
   - Log in to your account and copy your personal authentication token.
   - o	To test the website after one session go to identity & access section in the ngrok left side menu and select authtoken and add new authtoken for next session.
   - In Colab, run the following command and paste your token when prompted:
     ```python
     import getpass
     print("Enter your ngrok authtoken:")
     authtoken = getpass.getpass()
     !ngrok authtoken $authtoken
     ```

4. **Run the Flask Application**
   - Execute the script to initialize the chatbot and set up the server.
   - The `ngrok` tunnel will provide a public URL for accessing the chatbot.

5. **Interact with the Chatbot**
   - Open the `ngrok` URL in your browser.
   - Use the user-friendly interface to type or speak queries in your preferred language.

6. **Stop the Application**
   - Terminate the Colab runtime once done to stop the server and release resources.

### Notes
- Ensure you have a stable internet connection while running the application.
- If you face issues with `ngrok` or the server, restart the runtime and repeat the steps.

## Project Overview
This chatbot aims to bridge the gap in healthcare information by providing multilingual, accessible, and accurate responses to maternity-related queries. Its implementation demonstrates how AI can empower communities by addressing language barriers and enhancing the dissemination of critical health information.

