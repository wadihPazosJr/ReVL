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
from langchain_core.messages import AIMessage, HumanMessage

# from utils import run_agent

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


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
    """Types out the specified text. This should be used when writing text in an input field or text editor, or when writing long blocks of code."""
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


@tool
def run_command(keys):
    """Given a sequence of keys, presses them in order. For example: ['command', 'space']"""
    for key in keys:
        key_down(key)

    for key in keys:
        key_up(key)


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
    run_command,
]


# llm_with_tools = llm.bind_tools(tools)

prompt_message = """"You are a very powerful AI agent that can interact with the GUI of a Mac computer.
You can move the mouse, click, press keys, write text, and open the spotlight. You can use these tools to perform tasks on the computer.
You will be given a state which describes the current state of the GUI, and you will need to perform a task on the GUI to change the state.
You will be given a task to perform on the GUI, and you will need to use the tools to complete the task.
You should take your task literally, and take into consideration the context of the state you are in. 
You should always prefer using the keyboard versus the mouse and clicking when possible. Thus, you should search for commands that can be done with the keyboard first, before defaulting to the mouse."""

MEMORY_KEY = "chat_history"

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", prompt_message),
#         MessagesPlaceholder(variable_name=MEMORY_KEY),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )


class GUIAgent:
    def __init__(self, llm, verbose=True):
        self.llm_with_tools = llm.bind_tools(tools)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_message),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.chat_history = []
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=tools, verbose=verbose
        )

    def run(self, input_text):
        result = self.agent_executor.invoke(
            {"input": input_text, "chat_history": self.chat_history}
        )
        self.chat_history.extend(
            [HumanMessage(content=input_text), AIMessage(content=result["output"])]
        )

        return result["output"]


# GUIInteractionAgentExecutor = AgentExecutor(agent=GUIAgent, tools=tools, verbose=True)

# out = list(
#     agent_executor.stream(
#         {"input": "Open the application 'arc' and click on the coordinate (2071, 136)"}
#     )
# )

# out = list(
#     agent_executor.stream(
#         {
#             "input": "Open the application 'messages', then click on the coordinate (2710, 830), then write 'Sending you this with my AI!' and press enter."
#         }
#     )
# )


# input_text = "Open the application 'messages', then click on the coordinate (2710, 830), then write 'Sending you this with my AI!' and press enter."

# out = run_agent(agent_executor, input_text)
