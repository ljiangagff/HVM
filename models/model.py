import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import einsum
from einops import rearrange
from . import (
    config,
    decoder,
    encoders,
    CVT,
    CMT,
)
from .positionEncoding import PositionalEncoding
PAD_TOKEN = config.PAD_TOKEN
START_TOKEN = config.START_TOKEN
END_TOKEN = config.END_TOKEN
MAX_SEQ_LENGTH = config.MAX_SEQ_LENGTH

'''
The task here is a seq2seq task which translates a formula image encoding to a latex sequence
'''


def check_scale(hdim, encoder_type):
    if encoder_type != 0:
        hdim = 256
    size = {
        192: "Small",
        256: "Base",
        320: "Large"
    }
    if hdim not in size:
        print("Wrong scale, set to default 256")
        hdim = 256
    scale = size[hdim]
    return scale


def encoder_initiator(encoder_type, encoder_params, device, scale='Base'):
    if encoder_type == 0:
        print(f"Current CVT encoder size {scale}")
        init_CVT = getattr(CVT, f'CVT_{scale}')
        image_encoder = init_CVT(encoder_params)

    elif encoder_type == 1:
        image_encoder = encoders.StemTransformerEncoder()
        print('Applying encoder StemTransformer')
    elif encoder_type == 2:
        image_encoder = encoders.VisionTransformer()
        print('Applying encoder Vision Transformer')
    elif encoder_type == 3:
        image_encoder = encoders.DenseNet()
        print('Applying encoder DenseNet')
    elif encoder_type == 4:
        image_encoder = CMT.init_CMT()
        print('Applying encoder CMT')
    else:
        image_encoder = CVT.CVT_Base(encoder_params)

    return image_encoder.to(device)


def text_encoder_initiator(text_encoder_type, device, vocab_embedding, scale='Base'):
    cfg = config.TextEncoderConfig[scale]
    if text_encoder_type == 0:
        text_encoder = decoder.TextEncoder(device=device,
                                           embeddingLayer=vocab_embedding,
                                           dim=cfg['dim'],
                                           heads=cfg['nhead'],
                                           num_layers=cfg['num_layers'])
    else:
        text_encoder = decoder.NoneTextEncoder(device, vocab_embedding, cfg['dim'])
    return text_encoder.to(device)


def decoder_initiator(decoder_type, device, scale='Base', vocab_size=510):
    if decoder_type == 0: 
        cfg = config.FTransformerDecoderConfig[scale]
        text_decoder = decoder.FTransformerDecoder(
            decoder.FTransformerDecoderLayer(
                d_model=cfg['dim'],
                nhead=cfg['nhead'],
                dim_feedforward=cfg['dim']*4,
            ),
            num_layers=cfg['num_layers'],
            ).to(device)
    else:
        text_decoder = decoder.LSTMDecoderWithAttention(device, vocab_size)
    return text_decoder


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


class Model(nn.Module):
    def __init__(self, vocab, device,
                 hdim=256,
                 contrastive_loss_weight=0.5,
                 encoder_type=0,
                 text_encoder_type=0,
                 decoder_type=0,
                 encoder_params = {},
                 ):

        super(Model, self).__init__()
        self.hdim = hdim
        self.vocab = vocab
        self.device = device
        self.embedding = nn.Embedding(len(vocab), hdim)

        scale = check_scale(hdim, encoder_type)
        self.image_encoder = encoder_initiator(encoder_type, encoder_params, device, scale)
        self.text_encoder = text_encoder_initiator(text_encoder_type, device, self.embedding, scale)
        self.decoder= decoder_initiator(decoder_type, device, scale, vocab_size=len(self.vocab))
        self.decoder_type = decoder_type

        # No text encoder, just nn embedding and pos embedding
        if text_encoder_type != 0:
            contrastive_loss_weight = 0.0
        self.contrastive_loss_weight = contrastive_loss_weight
        self.caption_loss_weight = 1.0 - self.contrastive_loss_weight
        self.alpha = encoder_params['alpha']
        print(f'contrast loss weight {self.contrastive_loss_weight}, caption_loss_weight {self.caption_loss_weight}, alpha {self.alpha}')

        self.text_cls_token = nn.Parameter(torch.randn(hdim))
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.image_to_latents = EmbedToLatents(hdim, hdim)
        self.text_to_latents = EmbedToLatents(hdim, hdim)
        # whether the encoder has initialized position encoding
        self.pos = PositionalEncoding(hdim)
        self.ce = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.classification_layer = nn.Linear(
            hdim, len(vocab)).to(device=device)

    def contrastive_loss(self, text, image):
        batch, device = text.shape[0], text.device
        sim = einsum('i d, j d -> i j', text, image)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)
        contrastive_loss = (self.ce(sim, contrastive_labels) +
                            self.ce(sim.t(), contrastive_labels)) * 0.5
        return contrastive_loss

    def forward(self, input, teaching_force):
        """This function should only be used for training
        Args:
            input (torch.Tensor): bsz, height, width, hdim; The raw data of images
            tf_tokens (torch.Tensor): bsz, output_len, hdim; The teach_forcing_tokens, truth for prediction
                Each tensor needs to start with start token and end with end token except for paddings

        Returns:
            (torch.Tensor): [description]
        """
        # Get encoding
        image_tokens, image_embed = self.image_encoder(input)
        text_tokens, text_embed = self.text_encoder(teaching_force)

        if self.contrastive_loss_weight:
            image_latents = self.image_to_latents(image_embed)
            text_latents = self.text_to_latents(text_embed)
            contrastive_loss = self.contrastive_loss(
                text_latents, image_latents)
            contrastive_loss = contrastive_loss * self.contrastive_loss_weight
        else:
            contrastive_loss = 0

        if self.caption_loss_weight:
            # [b, n, c] to [n, b, c]
            if self.decoder_type == 0:
                tgt = self.pos(text_tokens.permute(1, 0, 2))
            else:
                tgt = text_tokens.permute(1, 0, 2) 
            memory = image_tokens.permute(1, 0, 2)
            # decode the tokens, output_len, bsz, hdim
            output = self.decoder(tgt, memory)
            # output_len, bsz, vocab_size: (n, b, v)
            logits = self.classification_layer(output)
            logits = rearrange(logits, 'n b c -> b c n')
            # loss = criterion(outputs[:, :-1, :].permute(0, 2, 1), labels[:, 1:])
            caption_loss = self.ce(
                logits[:, :, :-1], teaching_force[:, 1:])
            caption_loss = caption_loss * self.caption_loss_weight
        else:
            caption_loss = 0

        return contrastive_loss + caption_loss

    # def beam_forward(self, input, teaching_force):
    #     memory = input.permute(1, 0, 2)
    #     text_tokens, _ = self.text_encoder(teaching_force)
    #     tgt = self.pos(text_tokens.permute(1, 0, 2))
    #     ouput = self.decoder(tgt, memory)
    #     logits = self.classification_layer(ouput)
    #     logits = rearrange(logits, 'n b c -> b n c')
    #     return logits

    def greedy_forward(self, input, teaching_force, cache):
        memory = input.permute(1, 0, 2)
        text_tokens, _ = self.text_encoder(teaching_force)
        tgt = self.pos(text_tokens.permute(1, 0, 2))
        output, cache = self.decoder(tgt, memory, cache)
        logits = self.classification_layer(output)
        logits = rearrange(logits, 'n b c -> b n c')
        return logits, cache


    def greedy_search_batch(
        self, 
        X, 
        predictions = MAX_SEQ_LENGTH+1,
    ):
        with torch.no_grad():
            device = X.device
            Y = torch.ones(X.shape[0], 1).to(device).long()
            cache = None
            for i in range(predictions):
                logits, cache = self.greedy_forward(X, Y, cache)
                next_probs = logits[:, -1].log_softmax(-1)
                max_next_probs, next_tokens = next_probs.max(-1)
                next_tokens = next_tokens.unsqueeze(-1)
                Y = torch.cat((Y, next_tokens), axis = 1)
        return Y
