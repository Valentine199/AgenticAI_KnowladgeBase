from crewai import Agent, Crew, Task, Process
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your OpenAI API key from environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" # Make sure this is set, or defined directly

# Initialize the LLM (using gpt-4o for better performance)
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Define the agents with specific roles and goals
extractor = Agent(
    role="Email Scanner",
    goal="Find all meetings, reminders, and tasks from the given emails, accurately extracting details like time, date, and subject.",
    backstory="You are an expert at scanning emails for key information. You meticulously extract every relevant detail.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

prioritizer = Agent(
    role="Schedule Optimizer",
    goal="Sort extracted items by urgency and time, preparing them for a daily agenda.",
    backstory="You are a master of time management, always knowing what needs to be done first. You organize tasks logically.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

formatter = Agent(
    role="Output Generator",
    goal="Generate a clean, polished, and concise daily agenda in bullet-point format, clearly listing all schedule items.",
    backstory="You are a professional secretary, ensuring all outputs are perfectly formatted and easy to read. You prioritize clarity.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Simulate email input
emails = """
1. Subject: Standup Call at 10 AM
2. Subject: Client Review due by 5 PM
3. Subject: Lunch with Sarah at noon
4. Subject: AWS Budget Warning – 80% usage
5. Subject: Dentist Appointment - 4 PM
"""

# Define the tasks for each agent
extract_task = Task(
    description=f"Extract all relevant events, meetings, and tasks from these emails: {emails}. Focus on precise details.",
    agent=extractor,
    expected_output="A list of extracted items with their details (e.g., '- Standup Call at 10 AM', '- Client Review due by 5 PM')."
)

prioritize_task = Task(
    description="Prioritize the extracted items by time and urgency. Meetings first, then deadlines, then other notes.",
    agent=prioritizer,
    context=[extract_task],  # The output of extract_task is the input here
    expected_output="A prioritized list of schedule items."
)

format_task = Task(
    description="Format the prioritized schedule into a clean, easy-to-read daily agenda using bullet points. Ensure concise language.",
    agent=formatter,
    context=[prioritize_task],  # The output of prioritize_task is the input here
    expected_output="A well-formatted daily agenda with bullet points."
)

# Instantiate the crew
crew = Crew(
    agents=[extractor, prioritizer, formatter],
    tasks=[extract_task, prioritize_task, format_task],
    process=Process.sequential,  # Tasks are executed sequentially
    verbose=2  # Outputs more details during execution
)

# Run the crew
result = crew.kickoff()
print("\n########################")
print("## Final Daily Agenda ##")
print("########################\n")
print(result)