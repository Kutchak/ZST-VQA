import cv2
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from helper import *

class VQADataset(Dataset):
  
  def question_to_int(self, sentences):
        return [self.vqa_vocab['<BOS>']] + [self.vqa_vocab[token] for token in self.tokenizer(sentences)]

  def answer_to_int(self, sentences):
        return [self.vqa_vocab[token] for token in self.tokenizer(sentences)] + [self.vqa_vocab['<EOS>']]

  def __init__(self, normal_split: bool, train_split: bool):
    
    normal_zero_path = "Normal_Split" if normal_split else "Zero_Split"
    train_test_path = "train" if train_split else "val"
    self.image_path = f"{normal_zero_path}/{train_test_path}/{train_test_path}2014/COCO_{train_test_path}2014_"
    question_json_path = f"{normal_zero_path}/{train_test_path}/v2_OpenEnded_mscoco_{train_test_path}2014_questions.json"
    answer_json_path = f"{normal_zero_path}/{train_test_path}/v2_mscoco_{train_test_path}2014_annotations.json"

    question_json = read_json(question_json_path)
    answer_json = read_json(answer_json_path)
    print("finished reading json")

    questions = list(prepare_questions(question_json))
    image_ids = list(prepare_images(question_json))
    answers = list(prepare_answers(answer_json))
    print("finished converting to list")

    self.tokenizer = get_tokenizer('basic_english')
    self.vqa_vocab = load_vocab()
    self.n_samples = len(questions) * 10

    print(self.n_samples)

    x = { "questions": [],"images": [] }
    y = { "answers": [] }

    for index in range(len(questions)):
        question = torch.tensor(self.question_to_int(questions[index]))
        img = image_ids[index]
        x["questions"].append(question)
        x["images"].append(img)
        for answer in answers[index]:
            y["answers"].append(torch.tensor(self.answer_to_int(answer)))

    print("finished preprocessing")

    x["questions"] = pad_sequence(x["questions"], padding_value=self.vqa_vocab['<PAD>'], batch_first=True)
    y["answers"] = pad_sequence(y["answers"], padding_value=self.vqa_vocab['<PAD>'], batch_first=True)

    print("finished padding")
    self.x = x
    self.y = y


  def __getitem__(self, index):
    return self.x["questions"][index // 10], cv2.resize(cv2.imread(f'{self.image_path}{str(self.x["images"][index // 10]).zfill(12)}.jpg'),(224,224)), self.y["answers"][index]

  def __len__(self):
    return self.n_samples
