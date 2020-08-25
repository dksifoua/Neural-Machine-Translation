# Neural Machine Translation

**Machine translation (MT)** is a sub-field of computational linguistics that investigates the use of software to translate a text from one natural language (such as French) to another (such as English). Solving this problem with artificial neural networks is often called **Neural Machine translation (NMT)**.

## Quick start

I used the `europarl-v7` parallel corpora to build models. The data is downloadable [here](http://www.statmt.org/europarl/v7/fr-en.tgz)!

```shell
> ./init.sh
```

## Modeling

### Encoder-Decoder architecture

The encoder-decoder architecture is a neural network design pattern. As shown in the figure below, the architecture is partitioned into two parts, the *encoder* and the *decoder*. The encoder's role is to encode the inputs into state, which often contains several tensors. Then the state is passed into the decoder to generate the outputs. In machine translation, the encoder transforms a source sentence, e.g., `Bonjour le monde !`, into state, e.g., a vector, that captures its semantic information. The decoder then uses this state to generate the translated target sentence, e.g., `Hello world!`.

<img src="./img/encoder-decoder.svg" alt="encoder-decoder architecture" />

### Sequence-to-Sequence model

The sequence-to-sequence model is based on the encoder-decoder architecture to generate a sequence output for a sequence input, as demonstrated below. Both the encoder and the decoder commonly use recurrent neural networks (RNNs) to handle sequence inputs of variable length. The hidden state of the encoder is used directly to initialize the decoder hidden state to pass information from the encoder to the decoder.

In this project, I tried several sequence-to-sequence models with LSTMs, Attention mechanisms, CNNs and Transformers.

<img src="./img/seq2seq.svg" alt="sequence-to-sequence" />

- LSTM SeqToSeq

- LSTM SeqToSeq with Luong Attention

- LSTM SeqToSeq with Badhanau Attention

- Convolutional SeqToSeq

- Transformer

| Model | Train loss | Valid loss | Train top-5 acc (%) | Test top-5 acc (%) | Time per epoch |
|:------|:----------:|:----------:|:-------------------:|:------------------:|:--------------:|
| LSTM SeqToSeq | 2.641 | 3.043 | 10.105 | 9.466 | 04:53 |
| LSTM SeqToSeq (Luong Attn) |  |  |  |  |  |
| LSTM SeqToSeq (Badhanau Attn) |  |  |  |  |  |
| CONV SeqToSeq |  |  |  |  |  |
| Transformer |  |  |  |  |  |

### Evaluation with BLEU score

### Inference with Beam Search

### Error Analysis

# References

