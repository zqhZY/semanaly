# -*- coding:utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

with open("data/mobile_dataset_jieba.csv") as f, open("data/mobile_dataset_chars.csv", "w") as f2:
    skip = True
    for line in f:
        if skip:
            skip = False
            f2.write(line)
            continue
        tokens = line.strip().decode("utf-8").split(",")
        record = [tokens[0]]
        tmp_text = ""
        for char in tokens[1]:
            if char != " ":
                tmp_text += char + " "
        record.append(tmp_text.strip())
        record.append(tokens[2])
        record.append(tokens[3])
        f2.write(",".join(record) + "\n")


with open("data/mobile_dataset_jieba_test.csv") as f, open("data/mobile_dataset_chars_test.csv", "w") as f2:
    skip = True
    for line in f:
        if skip:
            skip = False
            f2.write(line)
            continue
        tokens = line.strip().decode("utf-8").split(",")
        record = [tokens[0]]
        tmp_text = ""
        for char in tokens[1]:
            if char != " ":
                tmp_text += char + " "
        record.append(tmp_text.strip())
        record.append(tokens[2])
        record.append(tokens[3])
        f2.write(",".join(record) + "\n")

with open("data/mobile_dataset_jieba_test_cleaned.csv") as f, open("data/mobile_dataset_chars_test.csv", "w") as f2:
    skip = True
    for line in f:
        if skip:
            skip = False
            f2.write(line)
            continue
        tokens = line.strip().decode("utf-8").split(",")
        record = [tokens[0]]
        tmp_text = ""
        for char in tokens[1]:
            if char != " ":
                tmp_text += char + " "
        record.append(tmp_text.strip())
        record.append(tokens[2])
        record.append(tokens[3])
        f2.write(",".join(record) + "\n")