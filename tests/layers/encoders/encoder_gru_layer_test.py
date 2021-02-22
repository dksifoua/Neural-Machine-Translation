import unittest
import numpy as np

import torch

from nmt.layers.encoders.encoder_gru_layer import EncoderBiGruLayer


class TestEncoderBiGruLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.vocab_size = 30
        self.embedding_size = 10
        self.hidden_size = 32
        self.n_layers = 4
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5

        self.encoder_layer = EncoderBiGruLayer(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                               hidden_size=self.hidden_size, n_layers=self.n_layers,
                                               embedding_dropout=self.embedding_dropout,
                                               recurrent_dropout=self.recurrent_dropout)

    def test_init_model(self):
        # If the dropouts are not in [0, 1).
        with self.assertRaises(ValueError):
            _ = EncoderBiGruLayer(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                  hidden_size=self.hidden_size, n_layers=self.n_layers,
                                  embedding_dropout=-0.1, recurrent_dropout=1.01)

    def test_load_embedding_weights(self):
        # If the shape of embedding weights doesn't match.
        with self.assertRaises(ValueError):
            embedding_weights = torch.FloatTensor(np.random.randn(12, 54))
            self.encoder_layer.load_embedding_weights(embedding_weights=embedding_weights)

        # Else
        embedding_weights = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.encoder_layer.load_embedding_weights(embedding_weights=embedding_weights)
        np.testing.assert_array_equal(embedding_weights.numpy(), self.encoder_layer.embeddings.weight.detach().numpy())

    def test_fine_tune_embedding_weights(self):
        self.encoder_layer.fine_tune_embedding_weights(fine_tune=True)
        for param in self.encoder_layer.embeddings.parameters():
            self.assertTrue(param.requires_grad)
        self.encoder_layer.fine_tune_embedding_weights(fine_tune=False)
        for param in self.encoder_layer.embeddings.parameters():
            self.assertFalse(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        with self.assertRaises(ValueError):
            _ = self.encoder_layer(
                input_sequences=torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size)),
                sequence_lengths=torch.randint(low=1, high=seq_len + 1, size=(batch_size + 1,))
            )
        encoder_outputs = self.encoder_layer(
            input_sequences=torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size)),
            sequence_lengths=torch.randint(low=1, high=seq_len + 1, size=(batch_size,))
        )
        self.assertEqual(encoder_outputs.size(), torch.Size([self.n_layers * 2, batch_size, self.hidden_size]))



