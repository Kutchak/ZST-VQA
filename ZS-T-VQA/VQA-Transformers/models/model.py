import torch
import torch.nn as nn
from transformers import BertModel, SwinModel

# Create the BertClassfier class
class VQATransformer(nn.Module):
    def __init__(self):
        super(VQATransformer, self).__init__()

        # Instantiate BERT model
        self.question_bert = BertModel.from_pretrained('bert-base-uncased')
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8,batch_first=True)
        self.image_question_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.answer_bert = BertModel.from_pretrained('bert-base-uncased')

        self.image_question_answer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Instantiate an one-layer feed-forward classifier
        D_in, H, D_out = 768, 128, 1
        self.answer_classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        
    def forward(self, question_input_ids, question_attention_mask, answer_input_ids, answer_attention_mask, image):
        # print()
        # print("in model")

        # print("question_input_ids",question_input_ids.shape)
        question_features = self.question_bert(input_ids=question_input_ids,attention_mask=question_attention_mask)[0]
        # print("question_features",question_features.shape)

        # print("image",image.shape)
        image_features = self.swin(image).last_hidden_state
        # print("image_features",image_features.shape)

        image_question_feature = self.image_question_decoder(tgt=question_features, memory=image_features)
        # print("image_question_feature",image_question_feature.shape)

        # print("answer_input_ids",answer_input_ids.shape)
        # print("answer_attention_mask",answer_attention_mask.shape)
        answer_features = []

        for i in range(4):
            answer_features.append(self.answer_bert(input_ids=answer_input_ids[:,i],attention_mask=answer_attention_mask[:,i])[0])

        answer_features = torch.stack(answer_features,0)
        # print("answer_features",answer_features.shape)

        image_question_answer_feature = []

        for i in range(4):
            image_question_answer_feature.append(self.image_question_answer_decoder(tgt=answer_features[i], memory=image_question_feature)[:,0,:])

        # print(len(image_question_answer_feature))
        image_question_answer_feature = torch.stack(image_question_answer_feature,1)
        # print("image_question_answer_feature",image_question_answer_feature.shape)

        outputs = []
        for i in range(4):
            outputs.append(self.answer_classifier(image_question_answer_feature[:,i]))
        
        outputs = torch.stack(outputs,1)
        # print("outputs",outputs.shape)
        outputs = outputs.reshape(-1,4)

        return outputs