from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from business_analyst.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class BusinessAnalystCrew():
	"""BusinessAnalyst crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def data_understanding_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['data_understanding_agent'],
			verbose=True
		)

	@agent
	def data_cleaning_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['data_cleaning_agent'],
			verbose=True
		)
	
	@agent
	def exploratory_analysis_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['exploratory_analysis_agent'],
			verbose=True
		)
	
	@agent
	def data_analysis_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['data_analysis_agent'],
			verbose=True
		)
	
	@agent
	def visualization_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['visualization_agent'],
			verbose=True
		)
	
	@agent
	def reporting_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_agent'],
			verbose=True
		)
	
	@task
	def data_understanding_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_understanding_task'],
			agent=self.data_understanding_agent()
		)

	@task
	def data_cleaning_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_cleaning_task'],
			agent=self.data_cleaning_agent(),
			output_file='report.md'
		)
	
	# Create the crew that will run the task
	@crew
	def crew(self) -> Crew:
		"""Creates the BusinessAnalyst crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)