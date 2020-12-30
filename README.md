# Multimodal Sentiment Analysis

Multimodal Sentiment Analysis(MSA) is part of Sentiment Analysis. MSA uses several modality like visual, speech with text. It helps more better prediction when using only text.

The core of MSA is how to concatenate(or fuse) different modalities. But it is so hard because they usually have different frequency. At previous research, apply different method like LSTM or Transformer. Especially, Co-Transformer got better results than more older models. However, it also has problems that each modality affects the results about same portion.

So, I suggest unified-transformer with inputs that concatenated between different modalities. I guesses it can solve problem in Co-Transformer.

## 1. Requirements

- python : 3.7
- pytorch : 1.5
- transformers : 2.8.0

If you get "Segment fault", you must downgrade sentencepiece to version 0.1.91.

## 2. Model

