# TODO train this for at least 20 epochs, 
import math
from typing import Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    """Base class for all models."""

    def _init_(self, vocabulary, embedding_dim, num_layers):
        super()._init_(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)


class ImageEncoder(BaseImageEncoder):
    def _init_(self, embedding_dim):
        super()._init_()

        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        self.embedding_dim = self.dino.embed_dim

        self.fc = nn.Sequential(
            nn.Linear(self.dino.embed_dim, embedding_dim),
            nn.ReLU()
        )

        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale: int = 1

        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)

        intermediate_layers = \
        self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        cls_token, patch_tokens = intermediate_layers[1], intermediate_layers[0]

        encoding = self.fc(cls_token)

        return encoding


class PositionalEncoding(nn.Module):

    def _init_(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super()._init_()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape `[seq_len, batch_size, embedding_dim]`
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def _init_(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super()._init_()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src      : [seq_len, batch_size]
            src_mask : [seq_len, seq_len]
        Returns:
            output   : [seq_len, batch_size, ntoken]
        """

        src = self.embedding(src)

        src = src * math.sqrt(self.d_model)

        src = self.pos_encoder(src)

        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))

        output = self.transformer_encoder(src, src_mask)

        output = self.linear(output)

        # [seq_len, batch_size, vocabulary_size] --->[batch_size, vocabulary_size, seq_len]
        output = output.permute(1, 2, 0)

        return output


class CaptionGenerator(BaseCaptionGenerator):
    def _init_(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super()._init_(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5))

        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

        self.transformer_decoder = TransformerModel(
            ntoken=vocabulary_size,
            d_model=self.embedding_dim,
            nhead=2,
            d_hid=self.hidden_dim,
            nlayers=self.num_layers,
        )

        self.image_projection = nn.Linear(self.embedding_dim, self.embedding_dim)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        """Forward method.
        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        ################################################################################################################################################

        # 1. pre-fed vectors encoded-image
        # 2. embeddings using tokenization like BERT
        # 3. positional embeddings instead of RNN
        # 4. global self-attention /causal
        batch_size, seq_len = caption_indices.size()

        encoded_image = self.image_projection(encoded_image).unsqueeze(0)

        caption_indices = caption_indices.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(caption_indices.device)

        logits = self.transformer_decoder(encoded_image, src_mask=tgt_mask)

        ################################################################################################################################################

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):

        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,
                                  caption_indices=predicted_index,
                                  hidden_state=output['hidden_state'])

        return caption_indices