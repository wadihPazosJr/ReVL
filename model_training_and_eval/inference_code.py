import torch
from PIL import Image


recurse_model = None
click_model = None


def get_quadrant(image, quadrant_idx):
    width, height = image.size
    if quadrant_idx == 0:  # Top-left
        return image.crop((0, 0, width // 2, height // 2))
    elif quadrant_idx == 1:  # Top-right
        return image.crop((width // 2, 0, width, height // 2))
    elif quadrant_idx == 2:  # Bottom-left
        return image.crop((0, height // 2, width // 2, height))
    elif quadrant_idx == 3:  # Bottom-right
        return image.crop((width // 2, height // 2, width, height))


def recursive_click(img_path, k=2):
    img = Image.open(img_path)
    for _ in range(k):
        with torch.no_grad():
            logits = recurse_model(img)
            quadrant_idx = torch.argmax(logits).item()
        img = get_quadrant(img, quadrant_idx)

    with torch.no_grad():
        return click_model(img)
