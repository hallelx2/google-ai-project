import os
import pandas as pd  # for data manipulation
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import CSVSearchTool

from tools.analyst import data_analyst
from tools.clean_data import preprocess_df

# Create a chat model using Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Use CrewAI API key if applicable


# Define your agents with roles and goals
def_analyst = Agent(
    role='Data Analyst',
    goal='Analyze and prepare business data for further insights',
    backstory=
    """You are a data analyst with expertise in wrangling and preparing business data for analysis""",
    verbose=True,
    allow_delegation=False,
    tools=[preprocess_df],  # Replace with data cleaning function
    llm=llm)

business_analyst = Agent(
    role='Business Analyst',
    goal='Use Gemini Pro to gain insights from the prepared data',
    backstory=
    """You are a business analyst with a strong understanding of financial and business concepts. You leverage advanced analytics tools to extract meaningful insights from data.""",
    verbose=True,
    allow_delegation=False,
    tools=[data_analyst],  # Replace with Gemini Pro analysis function
    llm=llm)

visionary = Agent(
    role='Visionary',
    goal='Think through the deeper implications of the analysis',
    backstory=
    """you are a visionary technologist with a keen eye for identifying emerging trends and predicting their potential impact on various industries. Your ability to think critically and connect seemingly disparate dots allows you to anticipate disruptive technologies and their far-reaching implications.""",
    verbose=True,
    allow_delegation=False,
    llm=llm)

writer = Agent(
    role='Senior editor',
    goal='Writes professional quality articles that are easy to understand',
    backstory=
    """You are a details-oriented senior editor at the Wall Street Journal known for your insightful and engaging articles. You transform complex concepts into factual and impactful narratives.""",
    verbose=True,
    llm=llm,
    allow_delegation=True)


# Define tasks for your agents
task1 = Task(
    description="""The user will upload a business data file (CSV, Excel). Please clean and prepare the data for further analysis. This may involve handling missing values, outliers, and data type conversions.""",
    expected_output="Cleaned and prepared pandas dataframe",
    agent=def_analyst)

task2 = Task(
    description="""Using the cleaned data from the Data Analyst, leverage Gemini Pro to gain insights into the business. Here are some potential areas to explore (user can specify their own):

    * Trends over time (e.g., sales trends, customer growth)
    * Correlations between variables (e.g., marketing spend vs. customer acquisition)
    * Comparisons between different segments (e.g., customer demographics, product categories)
    * Identify key metrics and KPIs

    Provide a summary of the key insights discovered from the analysis.""",
    expected_output="Summary of key insights from Gemini Pro analysis",
    agent=business_analyst)

task3 = Task(
    description="""Using the insights provided by the Business Analyst,  think through the deeper implications of the findings. Consider the following questions as you craft your response:

    * What are the potential opportunities or challenges revealed by the analysis?
    * How might these insights inform strategic decision-making for the business?
    * What are the broader implications for the industry or market?

    Provide a thoughtful analysis of the long-term implications of the findings.""",
    expected_output="Analysis of long-term implications of insights",
    agent=visionary)

task4 = Task(
    description="""Using the insights from all the previous agents, craft a professional and informative report targeted towards a business audience. Ensure the report is well-structured, easy to understand, and incorporates the key findings and implications identified throughout the analysis.

    Make sure to reference the data source (user-uploaded file) and avoid using overly technical jargon.""",
    expected_output="Professional business report with insights and implications",
    agent=writer)


# Instantiate your crew with a sequential process
crew = Crew(
    agents=[def_analyst, business_analyst, visionary, writer],
    tasks=[task1, task2, task3, task4],
    verbose=2)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
