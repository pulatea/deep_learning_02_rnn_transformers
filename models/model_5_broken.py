import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack

from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    """Base class for all models."""

    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        print("number of layers ", num_layers)

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

        hidden_size = 384
        # linear layer that maps DINOv2 output to embedding dimension
        self.fc = nn.Linear(hidden_size, embedding_dim)
        # activation function f(x) = max(0, x)
        self.relu = nn.ReLU()

        # freeze the DINOv2 backbone
        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale: int = 1

        # extracting image features from DINOv2
        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        outputs = self.dino(resized_image)

        # extracting the <CLS> token representation
        # TODO try cls_token = outputs [:, 0] - won't work --> can't multiply 1x256 and 384x128
        # output shape is 256 x 384
        cls_token = outputs[:, :]

        # encoding the extracted <CLS> token to the dimension of the embedding
        encoding = self.fc(cls_token)

        return self.relu(encoding)


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.to_logits = nn.Linear(embedding_dim, vocabulary_size)

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
        """Forward method.

        :param encoded_image: torch.tensor of the shape [batch_size, *] or None
        :param caption_indices: torch.tensor of the shape [batch_size, sequence_length] or None
        :param args: e.g., hidden state

        :return: output dict at least with 'logits' and 'indices' keys,
            where: logits is the torch.tensor of the shape [batch_size, vocabulary_size, sequence_length]
                   indices is the torch.tensor of the shape [batch_size, sequence_length]
        """
        embeddings = self.embedding(caption_indices)  # [batch_size, sequence_length, embedding_dim]

        # Concatenate image encoding with token embeddings
        if encoded_image is not None:
            encoded_image = encoded_image.unsqueeze(1).repeat(1, embeddings.size(1), 1)
            embeddings = torch.cat((encoded_image, embeddings), dim=-1)

        embeddings = embeddings.permute(1, 0, 2)  # Transformer expects sequence_length first
        output = self.transformer_encoder(embeddings)  # [sequence_length, batch_size, embedding_dim]
        output = output.permute(1, 0, 2)  # Back to batch_first format

        logits = self.to_logits(output)  # [batch_size, sequence_length, vocabulary_size]
        return {'logits': logits, 'indices': logits.argmax(dim=-1)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
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
