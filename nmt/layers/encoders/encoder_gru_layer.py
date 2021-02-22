import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderBiGruLayer(nn.Module):

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, n_layers: int, embedding_dropout: float,
                 recurrent_dropout: float):
        """
        Encoder Layer with bidirectional GRU.

        :param vocab_size: The size of the vocabulary.
        :type vocab_size: int.
        :param embedding_size: The embedding size.
        :type embedding_size: int.
        :param hidden_size: The number of neurons in the GRU.
        :type hidden_size: int.
        :param n_layers: The number of stacked GRU.
        :type n_layers: int.
        :param embedding_dropout: The embedding dropout. Must be in [0, 1).
        :type embedding_dropout: float.
        :param recurrent_dropout: The recurrent (GRU) dropout. Must be in [0, 1).
        :type recurrent_dropout: float.

        :raise ValueError: If embedding_dropout or recurrent_dropout is below 0 or above 1.0.
        """
        if not (0 <= embedding_dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('The embedding_dropout and recurrent_dropout must be between 0 and 1.0.')

        super(EncoderBiGruLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.recurrent_dropout = recurrent_dropout if n_layers > 1 else 0
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=n_layers, dropout=self.recurrent_dropout,
                          bidirectional=True)

    def load_embedding_weights(self, embedding_weights: torch.FloatTensor) -> None:
        """
        Load the pre-trained embedding weights.

        :param embedding_weights: The pre-trained embedding weights.
        :type embedding_weights: torch.FloatTensor[vocab_size, embedding_size].

        :raise ValueError: If the embeddings' dimensions don't match.
        """
        if embedding_weights.size() != torch.Size([self.vocab_size, self.embedding_size]):
            raise ValueError("The dimensions of embeddings don't match.")

        self.embeddings.weight = nn.Parameter(embedding_weights)

    def fine_tune_embedding_weights(self, fine_tune: bool = False) -> None:
        """
        Activate or deactivate the fine-tuning of embeddings weights.

        :param fine_tune: Whether or not to fine-tune.
        :type fine_tune: bool.
        """
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_sequences: torch.IntTensor, sequence_lengths: torch.IntTensor) -> torch.FloatTensor:
        """
        Feed the network.

        :param input_sequences: The input sequences.
        :type input_sequences: torch.IntTensor[seq_len, batch_size].
        :param sequence_lengths: The lengths of each input sequence.
        :type sequence_lengths: torch.IntTensor[batch_size,].

        :raise ValueError: if the batch_size of input_sequences and sequence_lengths doesn't match.

        :return: torch.FloatTensor[n_layers * 2, batch_size, hidden_size]
        """
        if input_sequences.size(-1) != sequence_lengths.size(-1):
            raise ValueError("The batch_size of input_sequences and sequence_lengths doesn't match.")

        embedded = self.embeddings(input_sequences)  # [seq_len, batch_size, embedding_size]
        embedded = self.embedding_dropout(embedded)  # [seq_len, batch_size, embedding_size]
        packed = pack_padded_sequence(embedded, sequence_lengths, enforce_sorted=False)
        _, h_state = self.gru(packed)  # [seq_len, batch_size, hidden_size * 2]
        return h_state
