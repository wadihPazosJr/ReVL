import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertTokenizer

VOCAB_SIZE = 10000
EMBEDDING_DIM = 1024
NHEAD = 8
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 512
MAX_SEQ_LENGTH = 256
NUM_CATEGORIES = 10000

class Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.cnn = CNN(hidden_size=EMBEDDING_DIM * 2, embedding_size=EMBEDDING_DIM)
        # self.transformer = TransformerEmbeddingModule(
        #     vocab_size=VOCAB_SIZE,
        #     embedding_dim=EMBEDDING_DIM,
        #     nhead=NHEAD,
        #     num_encoder_layers=NUM_ENCODER_LAYERS,
        #     dim_feedforward=DIM_FEEDFORWARD,
        #     max_seq_length=MAX_SEQ_LENGTH,
        # )
        self.transformer = BERTTransformerModule(device=device)
        self.fc = nn.Sequential(
            nn.Linear(EMBEDDING_DIM*2, EMBEDDING_DIM * 4),  # From original feature size to hidden size
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM * 4, NUM_CATEGORIES),  # From hidden size to desired embedding size
        )
        self.device = device
        self.to(device)

    def forward(self, image, text):
        image_embedding = self.cnn(image)
        text_embedding = self.transformer(text)
        # image_embedding: [batch_size, EMBEDDING_DIM]
        # text_embedding: [batch_size, EMBEDDING_DIM]
        # print(image_embedding.shape, text_embedding.shape)
        combined = torch.cat([image_embedding, text_embedding], dim=1)
        # combined: [batch_size, EMBEDDING_DIM * 2]
        output = self.fc(combined)
        # output: [batch_size, NUM_CATEGORIES]
        return output
    
class CNN(torch.nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(CNN, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Freeze all layers of ResNet-50
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False

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
    
class BERTTransformerModule(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', device='cpu'):
        super(BERTTransformerModule, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.bert.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.device = device
        
        # Optional: Add additional layers for custom tasks, e.g., classification
        # For example, adding a dropout and a linear layer for binary classification
        self.dropout = nn.Dropout(0.1)
        # Assuming the BERT base model, which has a hidden size of 768
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, EMBEDDING_DIM),
        )

    def forward(self, text):
        # The BERT model returns a tuple with various outputs. The first output
        # is what we're interested in: the sequence of hidden-states at the output of the last layer
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the last layer hidden-states
        last_hidden_state = outputs.last_hidden_state
        return self.fc(self.dropout(last_hidden_state[:, 0, :]))

