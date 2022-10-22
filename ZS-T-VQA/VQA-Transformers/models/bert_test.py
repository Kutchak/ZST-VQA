from transformers import AdamW, get_linear_schedule_with_warmup
from bert import BertClassifier
from transformers import BertTokenizer
import re


MAX_LEN = 50

sent = "hello this is my string"
# sent2 ="hello this is my thing thingyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

encoded_sent = tokenizer.encode_plus(
    text=text_preprocessing(sent),  # Preprocess sentence
    add_special_tokens=False,        # Add `[CLS]` and `[SEP]`
    max_length=MAX_LEN,                  # Max length to truncate/pad
    truncation=True,
    padding='max_length',       # Pad sentence to max length
    return_tensors='pt',           # Return PyTorch tensor
    return_attention_mask=True      # Return attention mask
    )
b_input_ids = encoded_sent["input_ids"]
b_attn_mask = encoded_sent["attention_mask"]


# encoded_sent2 = tokenizer.encode_plus(
#     text=text_preprocessing(sent2),  # Preprocess sentence
#     add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
#     max_length=MAX_LEN,                  # Max length to truncate/pad
#     pad_to_max_length=True,         # Pad sentence to max length
#     # return_tensors='pt',           # Return PyTorch tensor
#     return_attention_mask=True      # Return attention mask
#     )

print(encoded_sent)
# print(encoded_sent2)


# test1 = tokenizer.encode(sent, add_special_tokens=True)
# test2 = tokenizer.encode(sent2, add_special_tokens=True)

# print(test1)
# print(test2)
# Instantiate Bert Classifier

print(b_input_ids)
print(tokenizer.decode(b_input_ids.tolist()[0]))
print(b_attn_mask)
bert_classifier = BertClassifier(freeze_bert=False)


logits = bert_classifier(b_input_ids, b_attn_mask)
print(logits.shape)
# print(logits)