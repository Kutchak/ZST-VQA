import re
import json
import torch
from os.path import exists

def load_vocab():
  vocab_path = "vocab.pth"
  if (exists(vocab_path)):
    return torch.load(vocab_path)

def read_json(file_path):
  with open(file_path, 'r') as f:
    return json.load(f)

def prepare_questions(questions_json):
  _special_chars = re.compile('[^a-z0-9 ]*')
  questions = [q['question'] for q in questions_json['questions']]
  for question in questions:
    question = question.lower()[:-1]
    question = _special_chars.sub('', question)
    yield question

def prepare_images(questions_json):
  image_ids = [q['image_id'] for q in questions_json['questions']]
  for image_id in image_ids:
      yield image_id

def prepare_answers(answers_json):
  """ Normalize answers from a given answer json in the usual VQA format. """
  _comma_strip = re.compile(r'(\d)(,)(\d)')
  _punctuation_chars = re.escape(r':;/[]"{}()=+_-><@`,?!')
  _punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
  _punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))
  answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
  def process_punctuation(s):
    if _punctuation.search(s) is None:
      return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
      s = s.replace(',', '')
    s = _punctuation.sub('', s)
    s = s.strip()
    return s

  for answer_list in answers:
    yield list(map(process_punctuation, answer_list))