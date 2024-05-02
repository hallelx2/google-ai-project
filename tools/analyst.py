from crewai_tools import tools
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd


@tools("Data Analyst")
# how to get the response for the pdf file
def data_analyst(df: pd.DataFrame, user_question: str) -> str:
    """
    This function generates a response to a user's question using a pre-trained large language model (LLM) and a pandas DataFrame.
    It helps you get response to some of the questions that you may have as an agent. For examole if you ask it for the sum of 
    a column or the person with the highest contribution in a dataset, it gives you a response based on that. 

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
        llm = ChatGoogleGenerativeAI(model='gemini-pro')

        # Create a pandas dataframe agent using the LLM and the provided DataFrame
        agent_executor = create_pandas_dataframe_agent(llm=llm,
                                                        df=df,
                                                        agent_type='zero-shot-react-description')

        # Invoke the agent with the user's question and return the response
        response = agent_executor.run(user_question)
        return response

    except Exception as e:
        # Handle potential errors during LLM interaction
        print(f"Error generating response: {e}")
        raise  # Re-raise the exception for further handling