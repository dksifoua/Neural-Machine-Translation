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

| `Models`              | `Parameters` | `Loss`           | `Perplexity`       | `Top-5 accuracy (%)`  |
|:----------------------|:------------:|:----------------:|:------------------:|:---------------------:|
| `1. BiGRU`            | `8,501,115`  | `2.051 - 2.561`  | `7.779 - 12.952`   | `12.365 - 11.633`     |
| `2. BiGRU + Badhanau` | `9,091,711`  | `2.258 - 2.356`  | `9.567 - 10.554`   | `11.998 - 11.911`     |
| `3. BiGRU + Luong`    | `11,649,659` | `1.795 - 2.208`  | `6.019- 9.099`     | `13.200 - 12.372`     |
| `4. Convolution`      | `7,965,273`  | `1.462 - 1.619`  | `4.316 - 5.048`    | `9.227 - 14.742`      |
| `5. Transformer`      |              |                  |                    |                       |

**Evaluation results (`Validation - Test`)**

- BLEU score

| `Models`     | `beam_size=1`       | `beam_size=3`       | `beam_size=5`       |
|:--------------------|:-------------------:|:-------------------:|:-------------------:|
| `1. BiGRU`            | `20.742 - 21.143`     | `21.212 - 22.840`     | `22.081 - 22.797`     |
| `2. BiGRU + Badhanau` | `24.894 - 24.983`     | `25.701 - 26.597`     | `25.770 - 26.105`     |
| `3. BiGRU + Luong`    | `27.215 - 28.706`     | `29.321 - 29.918`     | `29.525 - 30.395`     |
| `4. Convolution`      |                     |                     |                     |
| `5. Transformer`      |                     |                     |                     |

- Inference time

| `Models`              | `beam_size=1`           | `beam_size=3`           | `beam_size=5`           |
|:----------------------|:-----------------------:|:-----------------------:|:-----------------------:|
| `1. BiGRU`            | `00min:12s - 00min:12s` | `01min:12s - 01min:10s` | `02min:20s - 02min:27s` |
| `2. BiGRU + Badhanau` | `00min:17s- 00min:17s`  | `01min:41s - 01min:38s` | `03min:12s - 03min:06s` |
| `3. BiGRU + Luong`    | `00min:18s - 00min:18s` | `01min:47s - 01min:44s` | `03min:21s - 03min:17s` |
| `4. Convolution`      |                       |                       |                       |
| `5. Transformer`      |                       |                       |                       |

# References

- [1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
- [2] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
- [3] Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
- [4] Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N. (2017). Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122.
- [5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).


