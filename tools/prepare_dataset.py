# -*- coding:utf-8 -*-
import xlrd
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


excel_path1 = "/home/zqh/mygit/semanaly/mobile_classify/data/guangxi.data.ok.target.machine.segged_0.xlsx"
excel_path2 = "/home/zqh/mygit/semanaly/mobile_classify/data/henan.data.ok.target.machine.segged_0.xlsx"
stopword_dict = "/home/zqh/mygit/semanaly/dataset/stop_words_ch.txt"

stop_dict = {}
with open(stopword_dict) as d:
    for word in d:
        stop_dict[word.strip("\n")] = 1


def rm_stopwords(words):
    # read stop word dict and save in stop_dict
    tmp_list = []  # save words not in stop dict
    for word in words:
        if word not in stop_dict:
            tmp_list.append(word)
    return tmp_list

data_set = {}


def get_data_from_excel_v2(excel_path, head, is_first=True):
    """
    get label data from excel
    :return: data_set dict
    """

    data = xlrd.open_workbook(excel_path)
    table = data.sheets()[0]

    text = ""

    uid = table.cell(0, 0).value
    first_class = table.cell(0, 1).value
    second_class = table.cell(0, 2).value

    with open("./data/mobile_dataset.csv", "a") as f1, open("./data/mobile_dataset_no_label.csv", "a") as f2:
        if is_first:
            f1.write("id,text,first_class,second_class\n")
        for i in range(1, table.nrows):
            if head in table.cell(i, 0).value:
                f1.write(uid + "," + text.strip() + "," + first_class + "," + second_class + "\n")
                f2.write(text.strip() + "\n")
                uid = table.cell(i, 0).value
                first_class = table.cell(i, 1).value
                second_class = table.cell(i, 2).value
                text = ""
            elif table.cell(i, 0).value != "":
                sentence = table.cell(i, 1).value
                if len(sentence) < 2:
                    continue
                # sentence = re.findall(ur"[\u4e00-\u9fa5]+", sentence)
                #words = jieba.cut("".join(sentence), cut_all=False)
                #words = rm_stopwords(words)
                text += sentence + ' '
        f1.write(uid + "," + text + "," + first_class + "," + second_class + "\n")
        f2.write(text + "\n")


get_data_from_excel_v2(excel_path1, "guangxi")
get_data_from_excel_v2(excel_path2, "henan", is_first=False)
