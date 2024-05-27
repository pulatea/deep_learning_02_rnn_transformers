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

        self.freeze()

    def freeze(self):
        for param in self.dino.parameters():
            param.requires_grad = False

    def forward(self, image):
        scale: int = 1

        resized_image = F.interpolate(image, size=(scale * 224, scale * 224), mode="bilinear", align_corners=False)
        outputs = self.dino.get_intermediate_layers(resized_image, n=1, reshape=True, return_class_token=True)[0]

        cls_token, patch_tokens = outputs[1], outputs[0]

        encoding = self.fc(cls_token)

        return encoding


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5)
        )

        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=8,  # Number of attention heads
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            batch_first=True
        )

        self.to_logits = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.vocabulary_size)

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
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        if encoded_image is not None:
            encoded_image = encoded_image.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        memory = self.transformer.encoder(embeddings) if encoded_image is None else self.transformer.encoder(
            encoded_image)

        tgt = self._get_embeddings(caption_indices=caption_indices)
        output = self.transformer.decoder(tgt, memory)

        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2)}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        """Generates caption indices like torch.tensor([1, 23, 5, 8, 2]).

        :param encoded_image: torch.tensor of the shape [1, *]
        :param sos_token_index: index of the "start of sequence" token (int)
        :param eos_token_index: index of the "end of sequence" token (int)
        :param max_length: maximum caption length (int)

        :return: caption indices (list of the length <= max_length)
        """
        # Initialize the list to store the generated caption indices
        caption_indices = [sos_token_index]

        # Prepare the hidden state using the encoded image
        hidden_state = self.init_hidden_state(encoded_image)

        # Start the generation process
        for _ in range(max_length):
            # Get the last generated token (the most recent token)
            last_token = torch.tensor([[caption_indices[-1]]], device=encoded_image.device)

            # Generate the embeddings for the last token
            embeddings = self._get_embeddings(encoded_image=None, caption_indices=last_token)

            # Perform a forward pass through the RNN using the embeddings
            output, hidden_state = self.rnn(input=embeddings, hx=hidden_state)

            # Calculate the logits for the current output
            logits = self.to_logits(output)
            logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

            # Get the index of the predicted token
            predicted_index = logits.argmax(dim=-2).item()

            # Append the predicted token to the caption indices
            caption_indices.append(predicted_index)

            # If the predicted token is the EOS token, stop the generation process
            if predicted_index == eos_token_index:
                break

        return caption_indices
