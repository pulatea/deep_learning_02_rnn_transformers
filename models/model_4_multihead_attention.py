import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    """Base class for all models."""

    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()

        # loading pre-trained DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # initializing the embedding_dim
        self.embedding_dim = self.dino.embed_dim

        self.fc = nn.Sequential(
            # linear layer that maps DINOv2 output to embedding dimension
            nn.Linear(self.dino.embed_dim, embedding_dim),
            # activation function f(x) = max(0, x)
            nn.ReLU()
        )

        # freeze the DINOv2 backbone
        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale = 1

        # extracting image features from DINOv2
        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        outputs = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        # extracting the <CLS> token representation
        cls_token, patch_tokens = outputs[1], outputs[0]

        # encoding the extracted <CLS> token to the dimension of the embedding
        encoding = self.fc(cls_token)

        return encoding


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers, num_heads=8):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            nn.Dropout(0.5)
        )

        # Initialize LSTM
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Initialize Multihead Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads)

        self.to_logits = nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        """Sets the requires_grad parameter to False for some model parameters."""
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
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        # Apply multihead attention
        if encoded_image is not None:
            key_padding_mask = None  # adjust as necessary for your application
            attn_output, _ = self.multihead_attention(embeddings, embeddings, embeddings, key_padding_mask=key_padding_mask)
            embeddings = attn_output + embeddings  # Residual connection

        output, hidden_state = self.rnn(input=embeddings, hx=hidden_state)
        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']

            caption_indices.append(predicted_index.item())
            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None, caption_indices=predicted_index, hidden_state=output['hidden_state'])

        return caption_indices
