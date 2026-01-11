import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

# Check environment variables
print(f"CREWAI_MODEL: {os.getenv('CREWAI_MODEL')}")
print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

model = os.getenv("CREWAI_MODEL", "openai/qwen2.5:7b")
print(f"Using model: {model}")

try:
    llm = LLM(model=model)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# Define a simple agent
agent = Agent(
    role='Tester',
    goal='Verify LLM connection',
    backstory='You are a test agent.',
    verbose=True,
    llm=llm
)

# Define a simple task
task = Task(
    description='Say "Hello, World!"',
    expected_output='Hello, World!',
    agent=agent
)

# Define the crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# Kickoff
try:
    result = crew.kickoff()
    print("\nSuccess! Result:")
    print(result)
except Exception as e:
    print("\nError during kickoff:")
    print(e)
