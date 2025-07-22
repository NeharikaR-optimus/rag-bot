import os
from dotenv import load_dotenv # Used to load environment variables from .env file
from langchain_openai import AzureChatOpenAI # The LangChain connector for Azure OpenAI chat models
from langchain_core.prompts import ChatPromptTemplate # For structuring the conversation prompt
from langchain_core.output_parsers import StrOutputParser # To parse the LLM's output into a string

# 1. Load environment variables
# This line looks for a .env file in the current directory and loads its contents
# as environment variables. This keeps your sensitive keys out of your code.
load_dotenv()

# 2. Configure Azure OpenAI Client with Environment Variables
# Retrieve the necessary credentials from environment variables.
# We'll use os.getenv() which is safe if the variable isn't set (returns None).
# It's good practice to add checks for missing variables.
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01") # Default if not specified

# Basic validation to ensure critical variables are set
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY]):
    raise ValueError("One or more Azure OpenAI environment variables are not set. "
                     "Please check your .env file: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY.")

print(f"Connecting to Azure OpenAI Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME} at {AZURE_OPENAI_ENDPOINT}")

# 3. Initialize the Language Model (LLM)
# This creates an instance of your Azure OpenAI chat model in LangChain.
# `temperature` controls creativity (lower = more focused, higher = more diverse).
# `streaming=True` is important for real-time interaction, printing words as they are generated.
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7,
    streaming=True
)
print("Azure OpenAI Chat Model initialized successfully.")

# 4. Define the Prompt Template
# A prompt template is how you instruct the LLM and define placeholders for input.
# ChatPromptTemplate allows for different "roles" in the conversation:
# - "system": Sets the overall behavior/persona of the AI.
# - "user": Represents the user's input.
# - "assistant": Represents the AI's previous responses (used in multi-turn, not strictly needed for this basic one).
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. Answer questions concisely and professionally."),
    ("user", "{user_input}") # {user_input} is a placeholder for the actual user query
])
print("Prompt template defined.")

# 5. Create a LangChain Chain
# A chain is a sequence of components linked together.
# Here, we chain: Prompt -> LLM -> Output Parser.
# The `|` operator is the preferred way to do this in LangChain Expression Language (LCEL).
# - The `prompt` takes the user's input and formats it.
# - The `llm` (Azure OpenAI model) processes the formatted prompt.
# - The `StrOutputParser` takes the raw output from the LLM and converts it into a simple string.
chain = prompt | llm | StrOutputParser()
print("LangChain pipeline (chain) created.")

# 6. Main Chat Loop
# This loop continuously takes user input, sends it to the chain, and prints the response.
def run_basic_chatbot():
    print("\n--- LangChain Basic Chatbot (powered by Azure OpenAI) ---")
    print("Type 'exit' to quit at any time.")
    print("---------------------------------------------------------")

    while True:
        user_input = input("You: ") # Get input from the user
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break

        print("Bot (streaming): ", end="") # Print "Bot (streaming): " and keep cursor on same line
        try:
            # Invoke the chain. The `invoke` method is for single requests.
            # For streaming, we use `stream`.
            # The input to the chain matches the placeholder in the prompt ({user_input}).
            for chunk in chain.stream({"user_input": user_input}):
                print(chunk, end="", flush=True) # Print each part of the response as it comes
            print() # Move to the next line after the full response is printed
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your Azure OpenAI deployment and API keys.")

# Ensure the chatbot runs when the script is executed
if __name__ == "__main__":
    run_basic_chatbot()