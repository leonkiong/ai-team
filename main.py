from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

search_tool = DuckDuckGoSearchRun()

#streamlit callback
def streamlit_callback(step_output):
    # This function will be called after each step of the agent's execution
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(
                    f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)


# Team Members
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover the best {topic}',
  verbose=True,
  memory=True,
  backstory="""Driven by curiosity, you're at the forefront of
    innovation, eager to explore and share knowledge that could change
    the world.""",
  tools=[search_tool],
  allow_delegation=True,
  step_callback=streamlit_callback,
)

writer = Agent(
  role='Writer',
  goal='Write a well balanced article on the best {topic}',
  verbose=True,
  memory=True,
  backstory="""With a flair for simplifying complex topics, you craft
    engaging narratives that captivate and educate, bringing new
    discoveries to light in an accessible manner.""",
  tools=[search_tool],
  allow_delegation=False,
  step_callback=streamlit_callback,
)

#Tasks

research_task = Task(
  description="""Identify the best {topic}.
    Focus on identifying pros and cons and the overall narrative.
    and how the {topic} benefits its user.""",
  expected_output='A comprehensive 3 paragraphs long report on the best {topic}.',
  tools=[search_tool],
  agent=researcher,
)

write_task = Task(
  description="""Compose an insightful article on {topic}.
    Focus on the pros and cons for the best {topic}, provide a list of the pros and cons
    This article should be easy to understand, engaging, and positive.""",
  expected_output='A 4 paragraph article on the {topic} formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

# Create Crew
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  verbose=2,
  manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),
  process=Process.hierarchical  # Optional: Sequential task execution is default
)

#Interface code

st.title("Team Leon-AI gives you the best recommendations")
st.subheader("This team comprises of an expert researcher, writer and manager", anchor=False, divider="rainbow")
user_prompt = st.text_input("Give me the best...")

if st.button("Generate") and user_prompt:
    with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
      with st.container(height=500, border=False):
        output = crew.kickoff(inputs={'topic': user_prompt})
      status.update(label="✅ Recommendation Ready!",
                      state="complete", expanded=False)

    st.subheader("Here is my recommendation", anchor=False, divider="rainbow")
    st.markdown(output)