# NLP data augmentation

This repository was created as part of my Master's Thesis on NLP at
Imperial College London. This project allows for three different
techniques for augmentation of text data

## Installation

### Python requirements
This project is written in python 3 and requires a python3 venv. Once
created, install the requirements: ```pip install -r
requirements.txt```

### Other dependencies

Most models require pre-trained word vector models. As these models
are relatively large, I ommitted them from the git repo. To download
the files automatically, run the shell script ```./pretrained_vectors.sh```


To download the pretrained vectors manually, save the following files
in the ```src/``` directory. 
- [Google News
Corpus](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
- extract to a ```.bin``` file
- [FastText Corpus](https://fasttext.cc/docs/en/english-vectors.html) -
download ```crawl-300d-2M.vec.zip``` and extract to a ```.vec``` file
- [GloVe](https://nlp.stanford.edu/projects/glove/) - download the
```glove.42B.300d.zip``` file and extract to a .txt file containing
the vectors (util for this in ```augment.py```). 




## Methods
### Threshold
Loads in a word embedding pre-trained on one of the large text corpora
##given above. Replaces the words in a sentence with their highest
##cosine similarity word vector neighbour that exceed a threshold
##given as an argument.

### POS-tag
Replaces all words of a given POS-tag (given as argument) in the
sentence with their most similar word vector from a large pre-trained
word embedding.

### Generative
Trains a two-layer LSTM network to learn the word representations of
given class. The network then generates new samples of the class by
initialising a random start word and following the LSTM's predictions
of the next word given the previous sequence. 


## Input

## Augmentating data



