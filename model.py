import torch
import torch.nn as nn
from torchvision import models

VOCAB_SIZE = 10000
EMBEDDING_DIM = 1024
NHEAD = 8
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 512
MAX_SEQ_LENGTH = 256
NUM_CATEGORIES = 10000

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN(hidden_size=EMBEDDING_DIM * 2, embedding_size=EMBEDDING_DIM)
        self.transformer = TransformerEmbeddingModule(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        self.fc1 = nn.Linear(EMBEDDING_DIM*2, EMBEDDING_DIM * 4)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(EMBEDDING_DIM * 4, NUM_CATEGORIES)

    def forward(self, image, text):
        image_embedding = self.cnn(image)
        text_embedding = self.transformer(text)
        # image_embedding: [batch_size, EMBEDDING_DIM]
        # text_embedding: [batch_size, EMBEDDING_DIM]
        print(image_embedding.shape, text_embedding.shape)
        combined = torch.cat([image_embedding, text_embedding[:, 0]], dim=1)
        # combined: [batch_size, EMBEDDING_DIM * 2]
        output = self.fc2(self.ReLU(self.fc1(combined)))
        # output: [batch_size, NUM_CATEGORIES]
        return output
    
class CNN(torch.nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(CNN, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Freeze all layers of ResNet-50
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Replace the FC layer of ResNet-50
        num_features = self.resnet50.fc.in_features  # Get the input feature size of the original FC layer
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, hidden_size),  # From original feature size to hidden size
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),  # From hidden size to desired embedding size
        )

    def forward(self, x):
        # Forward pass through the modified ResNet-50
        return self.resnet50(x)
    
class TransformerEmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerEmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
    def forward(self, text):
        # text: [batch_size, seq_length]
        embedded = self.embedding(text) + self.positional_encoding[:text.size(1), :]
        # embedded: [batch_size, seq_length, embedding_dim]
        transformer_output = self.transformer_encoder(embedded)
        # transformer_output: [batch_size, seq_length, embedding_dim]
        return transformer_output
