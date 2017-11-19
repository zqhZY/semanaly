# semanaly
semantic analysis using word2vector, doc2vector and other method. mainly for text similarity analysis.
[related link here](http://someth.duapp.com/2017/07/05/Word2vector%E5%8F%A5%E5%AD%90%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%AE%A1%E7%AE%97/)

## useage

### word2vector

#### data prepare
```
unzip dataset/me_train.zip
python read_data.py (in word2vector dir)
```

#### train for w2v and d2v
```
mkdir model
python word2vector.py (in word2vector dir)
python doc2vector.py (in word2vector dir)
```

#### test for text similarity use word2vector
```
python sample.py
python shottext.py
```

### lstm
```
cd lstm
python lstm.py
python shottext_lstm.py
```

### textclassfier
- demo text classfier using word2vector、cnn、lstm implemented by pytorch.
- kfold implemented for train

### tools
tools for data preprocess
