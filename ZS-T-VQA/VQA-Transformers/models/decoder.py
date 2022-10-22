import numpy as np
import torch.nn as nn
import torch

decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8,batch_first=True)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 40, 768)
tgt = torch.rand(10, 49, 768) # change middle later 196
image_question_feature = transformer_decoder(tgt, memory)

# print(transformer_decoder)
print()
print(image_question_feature.shape)
# print(out)
ans_feature_list = []

# for i in range(4)
#     ans_feature_list.append(model(image_question_feature,mc[i]))

# return ans_feature_list

# loss(answer,targets)