import torch
import torch.nn as nn
import torch.nn.functional as F


class BadhanauAttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, n_layers: int):
        """
        Attention Mechanism: Badhanau style.

        :param hidden_size: The number of neurons in the GRU.
        :type hidden_size: int.
        :param n_layers: The number of stacked GRU.
        :type n_layers: int.
        """
        super(BadhanauAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.fc = nn.Linear(n_layers, 1)
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def combine_hidden_state(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Combine the hidden state of different layers of the encoder (only if the encoder has more than one layer).

        :param hidden_state: The hidden state returned by the encoder.
        :type hidden_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
        :return: torch.FloatTensor[1, batch_size, hidden_size]
        """
        if self.n_layers > 1:
            hidden_state = hidden_state.permute(1, 2, 0)  # [batch_size, hidden_size, n_layers]
            hidden_state = self.fc(hidden_state)  # [batch_size, hidden_size, 1]
            hidden_state = hidden_state.permute(2, 0, 1)  # [1, batch_size, hidden_size]
        return hidden_state

    def forward(self, hidden_state: torch.FloatTensor, encoder_outputs: torch.FloatTensor,
                sequence_mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        :param hidden_state: The hidden state returned by the encoder.
        :type hidden_state: torch.FloatTensor[n_layers, batch_size, hidden_size].
        :param encoder_outputs: The encoder outputs.
        :type encoder_outputs: torch.FloatTensor[seq_len, batch_size, hidden_size].
        :param sequence_mask: The mask to not take into account pad tokens in the source sequence during the attention
            computation.
        :type sequence_mask: torch.BoolTensor[seq_len, batch_size, 1]. Default: None.
        :return: torch.BoolTensor[seq_len, batch_size, 1]
        """
        if not (hidden_state.shape[-1] == encoder_outputs.shape[-1] == self.hidden_size):
            raise ValueError('Hidden size does not match in encoder_outputs and hidden_state.')

        hidden_state = self.combine_hidden_state(hidden_state=hidden_state)  # [1, batch_size, hidden_size]

        # Calculating the alignment scores
        scores = self.V(torch.tanh(
            self.W1(encoder_outputs) + self.W2(hidden_state))
        )  # [seq_len, batch_size, 1] alignment scores

        # Apply mask to ignore <pad> tokens
        if sequence_mask is not None:
            scores = scores.masked_fill(sequence_mask == 0, -1e18)

        # Calculating the attention weights by applying softmax to the alignments scores
        attention_weights = F.softmax(scores, dim=0)  # [seq_len, batch_size, 1]

        return attention_weights
