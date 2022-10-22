import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageEncoder, self).__init__()
        self.model = models.vgg19(pretrained=True)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, image):
        with torch.no_grad():
            features = self.model(image)
        features = self.fc(features)
        l2_norm = F.normalize(features, p=2, dim=1).detach()
        return l2_norm

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, feature_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_dim, feature_size)

    def forward(self, text):
        embedding = self.embedding(text)
        embedding = self.tanh(embedding)
        embedding = embedding.transpose(0,1)
        _, (hidden, cell) = self.lstm(embedding)
        text_feature = torch.cat((hidden, cell), dim=2)
        text_feature = text_feature.transpose(0,1)
        text_feature = text_feature.reshape(text_feature.size()[0], -1)
        text_feature = self.fc(text_feature)
        return text_feature

class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, feature_size):
        super(CNNLSTM, self).__init__()

        # setup image and text encoders
        self.image_encoder = ImageEncoder(feature_size)
        self.question_encoder = TextEncoder(embedding_dim, hidden_dim, vocab_size, num_layers, feature_size)
        self.answer_encoder = TextEncoder(embedding_dim, hidden_dim, vocab_size, num_layers, feature_size)

        # dropout and fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(feature_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, image, question, answer):
        # grab image and text features
        image_feature = self.image_encoder(image)
        question_feature = self.question_encoder(question)
        iqa_features = []
        for i in range(len(answer[0])):
            iqa_features.append(image_feature * question_feature * self.answer_encoder(answer[:,i]))

        # combine image and text features
        # image_text_features = image_feature * question_feature
        for i in range(len(iqa_features)):
            iqa_features[i] = self.tanh(iqa_features[i])
            iqa_features[i] = self.fc1(iqa_features[i])
            iqa_features[i] = self.dropout(iqa_features[i])
            iqa_features[i] = self.tanh(iqa_features[i])
            iqa_features[i] = self.fc2(iqa_features[i])
        iqa_features = torch.stack(iqa_features, dim=1).reshape(-1,4)
        return iqa_features
