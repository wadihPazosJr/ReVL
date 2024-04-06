import pyautogui as gui

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


class MoveMouseInput(BaseModel):
    x: int = Field(description="The x coordinate to move the mouse to")
    y: int = Field(description="The y coordinate to move the mouse to")


@tool(args_schema=MoveMouseInput)
def move_mouse(x, y):
    """Move the mouse to the specified coordinates"""
    gui.moveTo(x, y)


@tool
def click():
    """Click the mouse at the current mouse position"""
    gui.click()


class PressKeyInput(BaseModel):
    key: str = Field(description="The key to press on the keyboard")


@tool(args_schema=PressKeyInput)
def press_key(key):
    """Press the specified key on the keyboard"""
    gui.press(key)


class KeyDownInput(BaseModel):
    key: str = Field(description="The key to press and hold on the keyboard")


@tool(args_schema=KeyDownInput)
def key_down(key):
    """Press and hold the specified key on the keyboard"""
    gui.keyDown(key)


class KeyUpInput(BaseModel):
    key: str = Field(description="The key to release on the keyboard")


@tool(args_schema=KeyUpInput)
def key_up(key):
    """Release the specified key on the keyboard"""
    gui.keyUp(key)


class WriteInput(BaseModel):
    text: str = Field(description="The text to write")


@tool(args_schema=WriteInput)
def write(text):
    """Write the specified text"""
    gui.write(text)


# @tool
# def get_screen_size():
#     return gui.size()


# @tool
# def get_mouse_position():
#     return gui.position()


# @tool
# def get_screenshot():
#     return gui.screenshot()


@tool
def open_spotlight(query):
    """Opens the spotlight search bar on Mac, types the query and presses enter"""
    key_down("command")
    press_key("space")
    key_up("command")
    write(query)
    press_key("enter")


tools = [
    move_mouse,
    click,
    # move_and_click,
    press_key,
    key_down,
    key_up,
    write,
    # get_screen_size,
    # get_mouse_position,
    # get_screenshot,
    open_spotlight,
]

llm_with_tools = llm.bind_tools(tools)

prompt_message = """"You are a very powerful AI agent that can interact with the GUI of a Mac computer.
You can move the mouse, click, press keys, write text, and open the spotlight. You can use these tools to perform tasks on the computer.
You will be given a task to perform on the GUI, and you will need to use the tools to complete the task."""

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

# out = list(
#     agent_executor.stream(
#         {"input": "Open the application 'arc' and click on the coordinate (2071, 136)"}
#     )
# )

out = list(
    agent_executor.stream(
        {
            "input": "Open the application 'messages', then click on the coordinate (2710, 830), then write 'Sending you this with my AI!' and press enter."
        }
    )
)

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
