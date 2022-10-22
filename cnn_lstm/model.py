import torch
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

class QuestionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, feature_size):
        super(QuestionEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_dim, feature_size)

    def forward(self, question):
        embedding = self.word_embeddings(question)
        embedding = self.tanh(embedding)
        embedding = embedding.transpose(0,1)
        _, (hidden, cell) = self.lstm(embedding)
        question_feature = torch.cat((hidden, cell), dim=2)
        question_feature = question_feature.transpose(0,1)
        question_feature = question_feature.reshape(question_feature.size()[0], -1)
        question_feature = self.tanh(question_feature)
        question_feature = self.fc(question_feature)
        return question_feature

    # def forward(self, question):
        # embedding = self.word_embeddings(question)
        # embedding = self.tanh(embedding)
        # lstm_out, _ = self.lstm(embedding.view(len(question), 1, -1))
        # lstm_out = self.tanh(lstm_out)
        # tag_space = self.fc(lstm_out.view(len(question), -1))
        # return tag_space

class VQA_CNN_LSTM(nn.Module):
    def __init__(self, question_vocab_size, answer_vocab_size, embedding_dim, hidden_dim, num_layers, feature_size):
        super(VQA_CNN_LSTM, self).__init__()

        # setup image and text encoders
        self.image_encoder = ImageEncoder(feature_size)
        self.question_encoder = QuestionEncoder(embedding_dim, hidden_dim, question_vocab_size, num_layers, feature_size)

        # dropout and fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        print("creating vqacnnlstm model")
        print("feature size", feature_size)
        print("anser vocab size", answer_vocab_size)
        self.fc1 = nn.Linear(feature_size, answer_vocab_size)
        self.fc2 = nn.Linear(answer_vocab_size, answer_vocab_size)

    def forward(self, image, question):
        # grab image and text features
        image_feature = self.image_encoder(image)
        question_feature = self.question_encoder(question)

        # combine image and text features
        image_text_features = image_feature * question_feature
        image_text_features = self.dropout(image_text_features)
        image_text_features = self.tanh(image_text_features)
        image_text_features = self.fc1(image_text_features)
        image_text_features = self.dropout(image_text_features)
        image_text_features = self.tanh(image_text_features)
        logits = self.fc2(image_text_features)
        return logits
