import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import streamlit as st
import pandas as pd

def upload_and_validate(accepted_file_types=["csv", "xlsx", "xls"]):
  """
  This function uploads a file, validates its type, and returns a pandas DataFrame.
  Args:
      accepted_file_types (list, optional): A list of accepted file type extensions (without the dot). Defaults to ["csv", "xlsx"].
  Returns:
      pandas.DataFrame: The uploaded data as a DataFrame, or None if an error occurs.
  """

  uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=accepted_file_types)

  if uploaded_file is not None:
    # Get the file type based on the extension
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type in accepted_file_types:
      if file_type == "csv":
        try:
          # Read the CSV file directly
          df = pd.read_csv(uploaded_file)
          return df
        except pd.errors.ParserError as e:
          st.error(f"Error parsing CSV file: {e}")
          return None
      else:  # file_type == "xlsx or "xls"
        try:
          # Read the Excel file and convert it to a DataFrame
          df = pd.read_excel(uploaded_file)
          return df
        except pd.errors.ParserError as e:
          st.error(f"Error parsing Excel file: {e}")
          return None
    else:
      st.warning(f"Unsupported file type: {e}")

  return None




# how to get the response for the pdf file
def get_response(df: pd.DataFrame, user_question: str,api_key, context= None) -> str:
    """
    This function generates a response to a user's question using a pre-trained large language model (LLM) and a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data for the LLM to reference.
        user_question (str): The user's question to be answered.

    Returns:
        str: The generated response from the LLM based on the user's question and the DataFrame data.

    Raises:
        Exception: If an error occurs during LLM interaction or response generation.
    """

    try:
        # Use ChatGoogleGenerativeAI specifying the Gemini Pro model
        llm = ChatGoogleGenerativeAI(model='gemini-pro', 
                                     google_api_key=api_key)

        # Create a pandas dataframe agent using the LLM and the provided DataFrame
        agent_executor = create_pandas_dataframe_agent(llm=llm,
                                                        df=df,
                                                        agent_type='zero-shot-react-description',
                                                        max_iterations=50)

        # Invoke the agent with the user's question and return the response
        response = agent_executor.run(input ={
                                            'question':f'{user_question}', 
                                              'context':f'{context}',
                                              'Personality':'You are a business data analyst that analyses business data alone\
                                                you check the data if it is a business data and perform analysis on the data as requested\
                                                    for by the user.'
                                              })
        return response

    except Exception as e:
        # Handle potential errors during LLM interaction
        print(f"Error generating response: {e}")
        raise  # Re-raise the exception for further handling


def main():
    load_dotenv()
    st.set_page_config(page_title = 'Business Analyser')
    st.header('Query your Businesses\'s data :chat')

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, I can help you gain insights from your data without you having much\
                      analytical or statistical knowledge. What would you love to know about \
                      about your business data?"),
        ]
    with st.sidebar:
        df = upload_and_validate()
        if df is not None:
            if isinstance(df, pd.DataFrame):
                st.success("File uploaded (CSV)!")
            else:  # data is a DataFrame (converted from Excel)
                st.success("File uploaded (Excel)!")
        else:
           print('Upload a file')
        context = st.text_input("Describe the dataset you are Submitting in few words")
        api_key = st.text_input('Put in your gemini API Key', type='password')


    # User functionality
    user_query = st.chat_input('Ask your question')

    if user_query is not None and user_query!='':
        response = get_response(df, user_query, api_key, context=context)
        st.session_state.chat_history.append(HumanMessage(content= user_query))
        st.session_state.chat_history.append(AIMessage(content = response))
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)


if __name__=='__main__':
    main()