import pyautogui as gui


class GUIUtil:
    def __init__(self):
        pass

    def move_mouse(self, x, y, duration=0):
        gui.moveTo(x, y, duration=duration)

    def click(self):
        gui.click()

    def move_and_click(self, x, y, duration=0):
        self.move_mouse(x, y, duration)
        self.click()

    def press_key(self, key):
        gui.press(key)

    def key_down(self, key):
        gui.keyDown(key)

    def key_up(self, key):
        gui.keyUp(key)

    def write(self, text, interval=0):
        gui.write(text, interval)

    def get_screen_size(self):
        return gui.size()

    def get_mouse_position(self):
        return gui.position()

    def get_screenshot(self):
        return gui.screenshot()


class GUIAgent(GUIUtil):
    def __init__(self):
        super().__init__()

    def open_spotlight(self):
        self.key_down("command")
        self.press_key("space")
        self.key_up("command")


# Usage
agent = GUIAgent()

screen_size = agent.get_screen_size()
print(screen_size)

mouse_position = agent.get_mouse_position()
print(mouse_position)

agent.open_spotlight()
agent.write("arc")
agent.press_key("enter")
agent.move_and_click(2071, 136)

screenshot = agent.get_screenshot()
screenshot.show()
