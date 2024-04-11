# import pyautogui as gui
import time

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from GUIInteractionAgent import GUIAgent as GUI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
gui = GUI(llm=llm, verbose=True)


class InteractGUIInput(BaseModel):
    task: str = Field(
        description="The task to complete. Should be a one step task such as opening an appliation, typing text, entering a commmand, etc."
    )
    state: str = Field(
        description="The current state of the GUI, such as what application is open and in focus. Should provide some context for the task."
    )


@tool(args_schema=InteractGUIInput)
def interact_gui(task, state=""):
    """Interacts with the GUI to complete the specified task. Should provide the current state of the GUI to provide context for the task. This state should be the app you are in, the overall task you are trying to complete, previous steps you have taken, etc."""
    input_text = f"Your current state is: {state}. Your task: {task}"
    out = gui.run(input_text)
    return out


tools = [interact_gui]

llm_with_tools = llm.bind_tools(tools)

OS = "macOS"

prompt_message = f""""You are a specialized personal AI assistant that lives within the user's personal computer that runs {OS}.
You have the ability to interact with the user's computer interface to perform tasks on their behalf.
You can open applications, move the mouse, click on the screen, write text, and more. Given a task from the user, 
you should first come up with a plan to use your tools to accomplish this plan, then execute the plan step by step.
Specifically, when interacting with the GUI, you should imagine it happening and do so one step at a time.

For example, 

User: Open Finder, and find the file named Tree of Thoughts.

You: Sure, I will first open Finder, and then I will open the search bar. Then I will search for the file named Tree of Thoughts.

- Interact with the GUI to open Finder.
- Then, interact with the GUI to open the search bar.
- Then interact with GUI to search for the file named Tree of Thoughts.

Your actions and plan should follow suit, meaning you should consider the state you are currently in, to complete the next task.

You should be very specific in your task of how you would like to interact with the GUI. For example, if I wanted to write a message in a chat application, I would say "Write 'Hello, World!' in the chat box you are currently in." or if I wanted to open a new tab in a browser, I would say "Open a new tab in the browser you are currently in."
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_message),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input_text = "Create a new file in visual studio code, write a python function for BFS'. Don't run or save it."

time.sleep(5)
out = list(agent_executor.stream({"input": input_text}))

# print("===========")

# print(out)


# Usage
# agent = GUIAgent()

# screen_size = agent.get_screen_size()
# print(screen_size)

# mouse_position = agent.get_mouse_position()
# print(mouse_position)

# agent.open_spotlight()
# agent.write("arc")
# agent.press_key("enter")
# agent.move_and_click(2071, 136)

# screenshot = agent.get_screenshot()
# screenshot.show()
