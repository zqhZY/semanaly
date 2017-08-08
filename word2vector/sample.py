import numpy as np


target = "./dataset/train_questions.txt"

rand_i = np.random.choice(range(36190),size=500,replace=False)
with open(target) as f, open("./dataset/target.txt", "w") as f2:
    count = 1
    for line in f:
        if count in rand_i:
            f2.write(line)
        count += 1

