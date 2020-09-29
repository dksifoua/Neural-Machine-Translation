# Neural Machine Translation

**Machine translation (MT)** is a sub-field of computational linguistics that investigates the use of software to translate a text from one natural language to another. Solving this problem with artificial neural networks is often called **Neural Machine translation (NMT)**.

In this project, I trained several sequence to sequence (seq2seq) models for Deutsch to English translation using PyTorch, TorchText and Spacy.

## Modeling

### Encoder-Decoder architecture

The encoder-decoder architecture is a neural network design pattern. As shown in the figure below, the architecture is partitioned into two parts, the *encoder* and the *decoder*. The encoder's role is to encode the inputs into state, which often contains several tensors. Then the state is passed into the decoder to generate the outputs. In machine translation, the encoder transforms a source sentence, into state, a vector, that captures its semantic information. The decoder then uses this state to generate the translated target sentence.

### Sequence-to-Sequence model

The sequence-to-sequence model is based on the encoder-decoder architecture to generate a sequence output for a sequence input, as demonstrated below. Both the encoder and the decoder commonly use recurrent neural networks (RNNs) to handle sequence inputs of variable length. The hidden state of the encoder is used directly to initialize the decoder hidden state to pass information from the encoder to the decoder.

In this project, I tried several sequence-to-sequence models with LSTMs, Attention mechanisms, CNNs and Transformers.

**Training results (`Train - Validation`)**

| SeqToSeq Models      | Loss                   | Perplexity             | Top-5 accuracy (%)         |
|:---------------------|:----------------------:|:----------------------:|:--------------------------:|
|BiGRU                 | 2.051 - 2.561          | 7.779 - 12.952         | 12.365 - 11.633            |
|BiGRU + Badhanau Attn | 2.258 - 2.356          | 9.567 - 10.554         | 11.998 - 11.911            |
|BiGRU + Luong Attn    | **1.795 - 2.208**      | **6.019- 9.099**       | **13.200 - 12.372**        |
|CONV2D + Dot Attn     |                        |                        |                            |
|Transformer           |                        |                        |                            |

**Evaluation results (`Validation - Test`)**

*TODO*

- [ ] Add inference time

| SeqToSeq Models      | `beam_size=1`       | `beam_size=3`       | `beam_size=5`       |
|:---------------------|:-------------------:|:-------------------:|:-------------------:|
|BiGRU                 | 20.742 - 21.143     | 21.212 - 22.840     | 22.081 - 22.797     |
|BiGRU + Badhanau Attn | 24.894 - 24.983     | 25.701 - 26.597     | 25.770 - 26.105     |
|BiGRU + Luong Attn    | **27.215 - 28.706** | **29.321 - 29.918** | **29.525 - 30.395** |
|CONV2D + Dot Attn     |                     |                     |                     |
|Transformer           |                     |                     |                     |

# References

