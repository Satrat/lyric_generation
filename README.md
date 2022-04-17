# Lyric Generation Using GRU
This repostitory includes training code, dataset and the pretrained model for the lyric generation stage of my computational creativity final project

Data was taken from this Kaggle dataset: https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset

And I used the following repository as a starting point for the training code:
https://github.com/PieroPaialungaAI/SongLyricsGenerator

## Directory Layout
### best_checkpoint
pretrained model

### data
lyrics dataset as csv file

### example.ipynb
demonstration of using pretrained model for inference

## Training Instuctions
Run the training script to re-parse the training data into lyrics.txt and train the model
```
python train.py
```
