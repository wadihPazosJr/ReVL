from GUIInteractionAgent import GUIAgent as GUI
from langchain_openai import ChatOpenAI
import time

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = GUI(llm=llm, verbose=True)
time.sleep(5)
agent.run("Open Visual Studio Code.")
