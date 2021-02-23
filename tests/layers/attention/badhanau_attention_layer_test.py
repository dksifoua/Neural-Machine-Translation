import unittest
import numpy as np

import torch

from nmt.layers.attention.badhanau_attention_layer import BadhanauAttentionLayer


class TestBadhanauAttentionLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_size = 16
        self.n_layers = 2
        self.attention_layer = BadhanauAttentionLayer(hidden_size=self.hidden_size, n_layers=self.n_layers)

    def test_forward(self):
        n_layers, batch_size, seq_len = 2, 16, 30
        with self.assertRaises(ValueError):
            h_state = torch.randn((n_layers, batch_size, self.hidden_size))
            enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size + 1))
            _ = self.attention_layer(hidden_state=h_state, encoder_outputs=enc_outputs)

        h_state = torch.randn((n_layers, batch_size, self.hidden_size))
        enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size))
        mask = torch.BoolTensor(np.random.randint(low=0, high=2, size=(seq_len, batch_size, 1)))
        attention_weights = self.attention_layer(hidden_state=h_state, encoder_outputs=enc_outputs, sequence_mask=mask)
        print(attention_weights.size())
        self.assertEqual(attention_weights.size(), torch.Size([seq_len, batch_size, 1]))
