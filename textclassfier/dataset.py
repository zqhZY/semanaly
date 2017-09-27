import random
from torchtext import data
from torchtext.data import Dataset
from sklearn.model_selection import KFold
import numpy as np


class MyDataset(data.TabularDataset):
    """

    """

    def splits(self, fields, dev_ratio=.1, shuffle=True, **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            fields: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
        """
        examples = self.examples
        if shuffle: random.shuffle(examples)

        dev_index = -1 * int(dev_ratio * len(examples))
        return (Dataset(fields=fields, examples=examples[:dev_index]),
                Dataset(fields=fields, examples=examples[dev_index:]))

    def kfold(self, k):
        """
        kfold using sklearn
        :param k:
        :return: index of kfolded
        """
        kf = KFold(k)
        examples = self.examples
        return kf.split(examples)

    def get_fold(self, fields, train_indexs, test_indexs, shuffle=True):
        """
        get new batch
        :return:
        """
        examples = np.asarray(self.examples)

        if shuffle: random.shuffle(examples)
        print list(train_indexs)
        return (Dataset(fields=fields, examples=examples[list(train_indexs)]),
                Dataset(fields=fields, examples=examples[list(test_indexs)]))
