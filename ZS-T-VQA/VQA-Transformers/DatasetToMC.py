from dataloader.helper import *
import random
import json
import re

normal_split = True
train_split = True

normal_zero_path = "Normal_Split" if normal_split else "Zero_Split"
train_test_path = "train" if train_split else "val"
image_path = f"{normal_zero_path}/{train_test_path}/{train_test_path}2014/COCO_{train_test_path}2014_"
question_json_path = f"{normal_zero_path}/{train_test_path}/v2_OpenEnded_mscoco_{train_test_path}2014_questions.json"
answer_json_path = f"{normal_zero_path}/{train_test_path}/v2_mscoco_{train_test_path}2014_annotations.json"

question_json = read_json(question_json_path)
answer_json = read_json(answer_json_path)
print("finished reading json")

questions = list(prepare_questions(question_json))
image_ids = list(prepare_images(question_json))
answers = list(prepare_answers_mc(answer_json))
print("finished converting to list")
print(len(answers))

set_answers = set()
ing_set = set()
prog = re.compile("[a-z]+[i][n][g][\s][a-z]*|[a-z]+[i][n][g]")

for i in range(len(answers)):
    set_answers.add(answers[i][0])
    if bool(prog.fullmatch(answers[i][0])):
        ing_set.add(answers[i][0])
# print(set_answers)

data = []

# for i in range(len(answers)):
#     answer_list = []
#     for j in range(10):
#         temp = []
#         while (len(set(temp)) != 4):
#             temp = answers[i] + random.sample(set_answers,3)
#         answer_list.append(temp)
#     data.append({"question": questions[i], "image_id": image_ids[i], "answers": answer_list})

yes_no = {"yes","no"}
set_numbers = {"0","1","2","3","4","5","6","7","8","9","10"}
directions_set = {"up", "down", "left", "right"}
directions_list = ["up", "down", "left", "right"]
colors_set = {"white", "black", "red", "orange", "yellow", "green", "blue", "indigo", "violet", "brown", "pink", "purple"}

for i in range(len(answers)):
    print(i)
    answer_list = []
    for j in range(10):
        temp = []
        while (len(set(temp)) != 4):
            if (answers[i][0] in yes_no):
                if (answers[i][0]== "yes"):
                    temp = answers[i] + random.sample(set_answers,3)
                    temp[1] = "no"
                else:
                    temp = answers[i] + random.sample(set_answers,3)
                    temp[1] = "yes"
            elif (answers[i][0].isnumeric()):
                temp = answers[i] + random.sample(set_numbers,3)
            elif (answers[i][0] in directions_set):
                temp = answers[i]
                for k in range(4):
                    if directions_list[k] not in temp:
                        temp.append(directions_list[k])
            elif (answers[i][0] in colors_set):
                temp = answers[i] + random.sample(colors_set,3)
            elif (answers[i][0] in ing_set):
                temp = answers[i] + random.sample(ing_set,3)
            else:
                temp = answers[i] + random.sample(set_answers,3)
        answer_list.append(temp)
    data.append({"question": questions[i], "image_id": image_ids[i], "answers": answer_list})

jsonString = json.dumps(data)


jsonFile = open("normal_train.json","w")
jsonFile.write(jsonString)
jsonFile.close()

# f = open("normal_train.json")

# data2 = json.load(f)
# print(len(data2))

# for i in data2[0:5]:
#     print(i)