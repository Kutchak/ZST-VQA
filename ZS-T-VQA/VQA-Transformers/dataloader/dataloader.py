import cv2
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AutoFeatureExtractor


from .helper import *
import os
import itertools
import matplotlib.pyplot as plt
from pprint import pprint
from random import randint
import numpy as np


class VQADataset(Dataset):

    def question_to_int(self, sentences):
        return torch.tensor([self.vqa_vocab['<BOS>']] + [self.vqa_vocab[token] for token in self.tokenizer(sentences)])

    def answer_to_int(self, sentences, max_length):
        pad_token = "<PAD>"
        word_list = [self.vqa_vocab[token]
                     for token in self.tokenizer(sentences)]
        current_length = len(word_list)
        add_length = max_length - current_length
        return torch.tensor(word_list + [self.vqa_vocab['<PAD>']]*(add_length) + [self.vqa_vocab['<EOS>']])

    def __init__(self, normal_split: bool, train_split: bool):
        # Load the BERT tokenizer
        tokenizerBert = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

        normal_zero_path = "Normal_Split" if normal_split else "Zero_Split"
        train_test_path = "train" if train_split else "val"
        self.image_path = f"{normal_zero_path}/{train_test_path}/{train_test_path}2014/COCO_{train_test_path}2014_"
        data_path = f"{normal_zero_path}/{train_test_path}/normal_{train_test_path}.json"

        # datapoints are already preprocessed in when creating the Multiple choice
        # length = 10
        datapoints = read_json(data_path)
        print("finished reading json")

        questions = [datapoint["question"]
                     for datapoint in datapoints]
        image_ids = [datapoint["image_id"]
                     for datapoint in datapoints]
        answers = [datapoint["answers"] for datapoint in datapoints]
        print("finished converting to list")

        # questions = list(prepare_questions(question_json))
        # image_ids = list(prepare_images(question_json))
        # answers = list(prepare_answers(answer_json))

        self.tokenizer = get_tokenizer('basic_english')
        self.vqa_vocab = load_vocab()
        self.n_samples = len(questions) * 10

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224")

        # BERT TOKENIZE QUESTIONS
        questions_input_ids = []
        questions_attn_mask = []

        for index in range(len(questions)):

            encoded_questions = tokenizerBert.encode_plus(
                text=questions[index],  # Preprocess questions
                add_special_tokens=False,        # Add `[CLS]` and `[SEP]`
                max_length=40,                  # Max length to truncate/pad
                truncation=True,
                padding='max_length',       # Pad sentence to max length
                return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
            )
            questions_input_ids.append(encoded_questions["input_ids"][0])
            questions_attn_mask.append(encoded_questions["attention_mask"][0])

        # BERT TOKENIZE ANSWERS
        answer_ids_final = []
        answer_attn_final = []

        for answer_list in range(len(questions)):
            answer_list_list = []
            answer_attn_mask_list = []
            for answer_set in range(10):
                answer_set_list = []
                answer_mask_list = []
                for answer_choice in range(4):
                    # print(answers[answer_list][answer_set][answer_choice])
                    encoded_questions = tokenizerBert.encode_plus(
                        # Preprocess questions
                        text=answers[answer_list][answer_set][answer_choice],
                        # Add `[CLS]` and `[SEP]`
                        add_special_tokens=True,
                        max_length=30,                  # Max length to truncate/pad
                        truncation=True,
                        padding='max_length',       # Pad sentence to max length
                        return_tensors='pt',           # Return PyTorch tensor
                        return_attention_mask=True      # Return attention mask
                    )
                    answer_set_list.append(encoded_questions["input_ids"][0])
                    answer_mask_list.append(
                        encoded_questions["attention_mask"][0])

                answer_list_list.append(answer_set_list)
                answer_attn_mask_list.append(answer_mask_list)
            answer_ids_final.append(answer_list_list)
            answer_attn_final.append(answer_attn_mask_list)

        # after
        x = {"questions_id": questions_input_ids,
             "questions_attention_mask": questions_attn_mask,
             "images": image_ids}
        y = {"answers_id": answer_ids_final,
             "answers_attention_mask": answer_attn_final}
        # print("length")
        # print(len(answer_ids_final))
        # print(answer_ids_final)
        # print("len2")
        # print(len(answer_attn_final))
        # print(answer_attn_final)

        self.x = x
        self.y = y

        # print(np.array(self.y["answers_id"]).shape)
        # print(np.array(self.y["answers_attention_mask"]).shape)

    def __getitem__(self, index):
        # for answer in self.y["answers"][index]:
        #   print([self.vqa_vocab.lookup_token(number) for number in answer])

        # print("index: ", index)
        # print("index divided by ", index//10)
        # print("index moded by ", index%10)
        # print("length of answers_id: ", len(self.y["answers_id"]))

        # print("translation: ",
        #    tokenizerBert.decode(self.y["answers_id"].tolist()[0]))

        # print("__getitem__")
        # print(self.y["answers_id"][index // 10][index % 10])

        # print(self.y["answers_id"])
        # print("\n\n\n\n")
        # print(self.y["answers_id"][index // 10][index % 10])
        shuffled_answers_id = self.y["answers_id"][index //
                                                   10][index % 10].copy()
        shuffled_answers_attention_mask = self.y["answers_attention_mask"][index //
                                                                           10][index % 10].copy()

        l, r = randint(1, 3), randint(1, 3)
        temp = shuffled_answers_id[r].clone()
        shuffled_answers_id[r] = shuffled_answers_id[l]
        shuffled_answers_id[l] = temp

        temp = shuffled_answers_attention_mask[r].clone()
        shuffled_answers_attention_mask[r] = shuffled_answers_attention_mask[l]
        shuffled_answers_attention_mask[l] = temp

        l, r = 0, randint(0, 3)
        temp = shuffled_answers_id[r].clone()
        shuffled_answers_id[r] = shuffled_answers_id[l]
        shuffled_answers_id[l] = temp

        temp = shuffled_answers_attention_mask[r].clone()
        shuffled_answers_attention_mask[r] = shuffled_answers_attention_mask[l]
        shuffled_answers_attention_mask[l] = temp

        target = r
        # print("Questions, ", len(self.x["questions_id"]))
        # print("images, ", len(self.x["images"]))

        image = cv2.imread(
            f'{self.image_path}{str(self.x["images"][index // 10]).zfill(12)}.jpg')
        # plt.imshow(image)
        # plt.show()

        inputs = self.feature_extractor(image, return_tensors="pt")
        # print(inputs["pixel_values"].shape)
        # print(image.size)
        # print("anser length",np.array(shuffled_answers_id).shape)
        # for i in range(4):
        #   print(shuffled_answers_id[i].shape)
        #   print()
        x1 = torch.Tensor(np.array(self.x["questions_id"][index // 10])).long()
        # print("x1",x1.shape)
        x2 = torch.Tensor(
            np.array(self.x["questions_attention_mask"][index // 10]))
        # print("x2",x2.shape)
        x3 = torch.Tensor(np.array(inputs["pixel_values"][0]))
        # print("x3",x3.shape)
        x4 = torch.stack(shuffled_answers_id, dim=0).numpy()
        # x4 = torch.Tensor([np.array(x) for x in shuffled_answers_id]).long()
        # print("x4",x4.shape)
        x5 = torch.stack(shuffled_answers_attention_mask, dim=0).numpy()
        # x5 = torch.Tensor([np.array(x) for x in shuffled_answers_attention_mask])
        # print("x5",x5.shape)
        x6 = target
        # print("x6",target)
        return x1, x2, x3, x4, x5, x6

    def __len__(self):
        return self.n_samples
