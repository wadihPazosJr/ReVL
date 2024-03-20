
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb as wb
from training_data_loader import get_data_loader

# Assuming the Model, CNN, TransformerEmbeddingModule classes are defined in model.py
from model import Model

# Hyperparameters
batch_size = 32
num_epochs = 1
learning_rate = 0.001

# Data loading and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = get_data_loader(batch_size=batch_size, transform=transform)

# Placeholder for text data loader
# In practice, you would need a dataset that yields both images and text
# text_train_loader = DataLoader(...)

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = Model(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Device configuration

def train(epoch):
    model.train()
    for batch_idx, (images, instructions, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, instructions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        wb.log({
            "iter": batch_idx + len(train_loader) * epoch,
            "train loss": loss.item(),
        })
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    wb.init(project="GUI-Grounding-Baseline")
    for epoch in range(num_epochs):
        train(epoch)
    torch.save(model.state_dict(), 'model.ckpt')
    print('Model saved')
    wb.finish()

