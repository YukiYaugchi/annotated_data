# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 17:27:16 2021.

@author: Keqi Deng (UCAS)
"""

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.contextual_block_encoder_layer import (
    ContextualBlockEncoderLayer,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import (
    make_pad_mask,  # noqa: H301
    get_activation,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import StreamPositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling_without_posenc import (
    Conv2dSubsamplingWOPosEnc,  # noqa: H301
)
from espnet2.asr.encoder.abs_encoder import AbsEncoder
import math
import torch
import logging
from typeguard import check_argument_types
from typing import (
    Optional,  # noqa: H301
    Tuple,  # noqa: H301
)

import numpy as np
np.random.seed(0)

class ContextualBlockParallelConformerEncoder(AbsEncoder):
    """Contextual Block Conformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
        block_size: block size for contextual block processing
        hop_Size: hop size for block processing
        look_ahead: look-ahead size for block_processing
        init_average: whether to use average as initial context (otherwise max values)
        ctx_pos_enc: whether to use positional encoding to the context vectors
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        pos_enc_class=StreamPositionalEncoding,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        block_size: int = 40,
        hop_size: int = 16,
        look_ahead: int = 16,
        init_average: bool = True,
        ctx_pos_enc: bool = True,
        streaming: bool = False,
        mask_rate: float = 0,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.pos_enc = pos_enc_class(output_size, positional_dropout_rate)
        activation = get_activation(activation_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
            )
            self.subsample = 1
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 3], strides=[2, 2]
            )
            self.subsample = 4
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size, output_size, dropout_rate, kernels=[3, 5], strides=[2, 3]
            )
            self.subsample = 6
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsamplingWOPosEnc(
                input_size,
                output_size,
                dropout_rate,
                kernels=[3, 3, 3],
                strides=[2, 2, 2],
            )
            self.subsample = 8
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
            self.subsample = 1
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
            self.subsample = 1
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
            self.subsample = 1
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: ContextualBlockEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate, streaming
                ),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                num_blocks,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # for block processing
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.init_average = init_average
        self.ctx_pos_enc = ctx_pos_enc
        
        assert self.look_ahead % self.hop_size == 0, "self.look_ahead % self.hop_size must be 0"
        if self.look_ahead // self.hop_size == 1:
            self.num_parallel = 1
        elif self.look_ahead // self.hop_size == 2:
            self.num_parallel = 2
        else:
            raise NotImplemented
        
        self.mask_rate = mask_rate

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        is_final=True,
        infer_mode=False,
        is_3steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
            infer_mode: whether to be used for inference. This is used to
                distinguish between forward_train (train and validate) and
                forward_infer (decode).
        Returns:
            position embedded tensor and mask
        """
        if self.training or not infer_mode:
            return self.forward_train(xs_pad, ilens, prev_states)
        else:
            return self.forward_infer(xs_pad, ilens, prev_states, is_final, is_3steps)

    def forward_train(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)

        # create empty output container
        total_frame_num = xs_pad.size(1)
        ys_pad = xs_pad.new_zeros(xs_pad.size())

        past_size = self.block_size - self.hop_size - self.look_ahead

        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        if self.block_size == 0 or total_frame_num <= self.block_size:
            xs_pad, masks, _, _, _, _, _ = self.encoders(
                self.pos_enc(xs_pad), masks, False, None, None
            )
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1)
            return xs_pad, olens, None

        # start block processing
        cur_hop = 0
        block_num = math.ceil(
            float(total_frame_num - past_size - self.look_ahead) / float(self.hop_size)
        )
        bsize = xs_pad.size(0)
        addin = xs_pad.new_zeros(
            bsize, block_num, xs_pad.size(-1)
        )  # additional context embedding vectors

        # first step
        if self.init_average:  # initialize with average value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).mean(1)
        else:  # initialize with max value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).max(1)
        cur_hop += self.hop_size
        # following steps
        while cur_hop + self.block_size < total_frame_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).max(1)
            cur_hop += self.hop_size
        # last step
        if cur_hop < total_frame_num and cur_hop // self.hop_size < block_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.hop_size, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).max(1)

        if self.ctx_pos_enc:
            addin = self.pos_enc(addin)

        xs_pad = self.pos_enc(xs_pad)

        # set up masks
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
#         mask_online.narrow(2, 1, self.block_size + 1).narrow(
#             3, 0, self.block_size + 1
#         ).fill_(1)
        
    
        rand_samp = np.random.rand()
        if self.num_parallel == 1:
            if rand_samp >= self.mask_rate:
                mask_online.narrow(2, 1, self.block_size + 1).narrow(
                    3, 0, self.block_size + 1
                ).fill_(1)
            else:
                mask_online.narrow(2, 1, self.block_size + 1).narrow(
                    3, 0, self.block_size + 1 - self.look_ahead   # masking look-ahead frames
                ).fill_(1)                 
        elif self.num_parallel == 2:            
            if rand_samp >= self.mask_rate:
                mask_online.narrow(2, 1, self.block_size + 1).narrow(
                    3, 0, self.block_size + 1
                ).fill_(1)
            else:
                #rand_samp = np.random.rand()
                if rand_samp >= self.mask_rate // 2:
                    mask_online.narrow(2, 1, self.block_size + 1).narrow(
                        3, 0, self.block_size + 1 - self.look_ahead   # masking look-ahead frames
                    ).fill_(1)
                else:
                    mask_online.narrow(2, 1, self.block_size + 1).narrow(
                        3, 0, self.block_size + 1 - (self.look_ahead // 2)   # masking look-ahead frames
                    ).fill_(1)
        else:
            raise NotImplemented

        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        )                        

        # fill the input
        # first step
        left_idx = 0
        block_idx = 0
        xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
            -2, left_idx, self.block_size
        )        
        
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
                -2, left_idx, self.block_size
            )                                                    
            
            left_idx += self.hop_size
            block_idx += 1
        # last steps
        last_size = total_frame_num - left_idx
        xs_chunk[:, block_idx, 1 : last_size + 1] = xs_pad.narrow(
            -2, left_idx, last_size
        )        

        # fill the initial context vector
        xs_chunk[:, 0, 0] = addin[:, 0]
        xs_chunk[:, 1:, 0] = addin[:, 0 : block_num - 1]
        xs_chunk[:, :, self.block_size + 1] = addin

        # forward
        ys_chunk, mask_online, _, _, _, _, _ = self.encoders(
            xs_chunk, mask_online, False, xs_chunk
        )

        # copy output
        # first step
        offset = self.block_size - self.look_ahead - self.hop_size + 1
        left_idx = 0
        block_idx = 0
        cur_hop = self.block_size - self.look_ahead
        
        ys_pad[:, left_idx:cur_hop] = ys_chunk[:, block_idx, 1 : cur_hop + 1]
        
        left_idx += self.hop_size
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            ys_pad[:, cur_hop : cur_hop + self.hop_size] = ys_chunk[
                :, block_idx, offset : offset + self.hop_size
            ]

            cur_hop += self.hop_size
            left_idx += self.hop_size
            block_idx += 1
        
        ys_pad[:, cur_hop:total_frame_num] = ys_chunk[
            :, block_idx, offset : last_size + 1, :
        ]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)

        olens = masks.squeeze(1).sum(1)                
        
        return (ys_pad, None), olens, None

    def forward_infer(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        is_final: bool = True,
        is_3steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        
        if prev_states is None:
            prev_addin = None            
            prev_block = None
            buffer_before_downsampling = None
            ilens_buffer = None
            buffer_after_downsampling = None
            n_processed_blocks = 0
            past_encoder_ctx = None
            past_encoder_ctx_r = None
            past_encoder_ctx_rr = None
        else:
            prev_addin = prev_states["prev_addin"]
            prev_block = prev_states["prev_block"]
            buffer_before_downsampling = prev_states["buffer_before_downsampling"]
            ilens_buffer = prev_states["ilens_buffer"]
            buffer_after_downsampling = prev_states["buffer_after_downsampling"]
            n_processed_blocks = prev_states["n_processed_blocks"]
            past_encoder_ctx = prev_states["past_encoder_ctx"]
            past_encoder_ctx_r = prev_states["past_encoder_ctx_r"]
            past_encoder_ctx_rr = prev_states["past_encoder_ctx_rr"]
        bsize = xs_pad.size(0)
        assert bsize == 1
        
#         if buffer_before_downsampling is not None:
#             logging.warning(f"buffer shape: {buffer_before_downsampling.shape}")
        
#         if is_final:
#             print("###FINAL###")
#         print("input:", xs_pad.shape)
        if prev_states is not None:  
#             print("buffer:", buffer_before_downsampling.shape)
            xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
            ilens += ilens_buffer
        else:
            buffer_before_downsampling = torch.zeros([1, 8, 80]).to(xs_pad.device)
            ilens_buffer = buffer_before_downsampling.size(1)
#             print("first buffer pad:", buffer_before_downsampling.shape)
            xs_pad = torch.cat([buffer_before_downsampling, xs_pad], dim=1)
            ilens += ilens_buffer
        
#         logging.warning(f"xs_pad shape: {xs_pad.shape}")

        if is_final:
            buffer_before_downsampling = None
        else:
            n_samples = xs_pad.size(1) // self.subsample - 1
            if n_samples < 2:
                next_states = {
                    "prev_addin": prev_addin,                    
                    "prev_block": xs_pad,
                    "buffer_before_downsampling": xs_pad,
                    "ilens_buffer": ilens,
                    "buffer_after_downsampling": buffer_after_downsampling,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                    "past_encoder_ctx_r": past_encoder_ctx_r,
                    "past_encoder_ctx_rr": past_encoder_ctx_rr,
                }
#                 logging.warning("n_sample<2")
                return (
                    (xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize, 0, self._output_size)),
                    xs_pad.new_zeros(bsize),
                    next_states,
                )

            n_res_samples = xs_pad.size(1) % self.subsample + self.subsample * 2
            buffer_before_downsampling = xs_pad.narrow(
                1, xs_pad.size(1) - n_res_samples, n_res_samples
            )
            xs_pad = xs_pad.narrow(1, 0, n_samples * self.subsample)
            

            ilens_buffer = ilens.new_full(
                [1], dtype=torch.long, fill_value=n_res_samples
            )
            ilens = ilens.new_full(
                [1], dtype=torch.long, fill_value=n_samples * self.subsample
            )

#         logging.warning(f"xs_pad shape: {xs_pad.shape}")
#         print("before emb:", xs_pad.shape)
        if isinstance(self.embed, Conv2dSubsamplingWOPosEnc):
            xs_pad, _ = self.embed(xs_pad, None)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)
#         print("after emb:", xs_pad.shape)

#         logging.warning(f"xs_pad shape: {xs_pad.shape}, {self.subsample}")
#         if buffer_after_downsampling is not None:
#             logging.warning(f"buffer after downsampling shape: {buffer_after_downsampling.shape}")
#         else:    
#             logging.warning(f"buffer after downsampling shape is None")
        # create empty output container
        if buffer_after_downsampling is not None:
#             print("buffer_after_downsampling:", buffer_after_downsampling.shape)
            xs_pad = torch.cat([buffer_after_downsampling, xs_pad], dim=1)
#             logging.warning(f"xs_pad shape: {xs_pad.shape}")
#         print("after emb2:", xs_pad.shape)

        total_frame_num = xs_pad.size(1)        
        if is_final:            
            past_size = self.block_size - self.hop_size - self.look_ahead
            block_num = math.ceil(
                float(total_frame_num - past_size - self.look_ahead)
                / float(self.hop_size)
            )            
            buffer_after_downsampling = None            
        else:
            if total_frame_num < self.block_size:
#             if total_frame_num <= self.block_size:
                next_states = {
                    "prev_addin": prev_addin,
                    "prev_block": xs_pad,
                    "buffer_before_downsampling": buffer_before_downsampling,
                    "ilens_buffer": ilens_buffer,
                    "buffer_after_downsampling": xs_pad,
                    "n_processed_blocks": n_processed_blocks,
                    "past_encoder_ctx": past_encoder_ctx,
                    "past_encoder_ctx_r": past_encoder_ctx_r,
                    "past_encoder_ctx_rr": past_encoder_ctx_rr,
                }
#                 logging.warning(f"total_frame_num < block_size: {total_frame_num}, {self.block_size}")
                return (
                    (xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize, 0, self._output_size), xs_pad.new_zeros(bsize, 0, self._output_size)),
                    xs_pad.new_zeros(bsize),
                    next_states,
                )

            overlap_size = self.block_size - self.hop_size
            block_num = max(0, xs_pad.size(1) - overlap_size) // self.hop_size
            res_frame_num = xs_pad.size(1) - self.hop_size * block_num
            buffer_after_downsampling = xs_pad.narrow(
                1, xs_pad.size(1) - res_frame_num, res_frame_num
            )
            xs_pad = xs_pad.narrow(1, 0, block_num * self.hop_size + overlap_size)
#             logging.warning(f"xs_pad shape: {xs_pad.shape}, {block_num}")
                    
        mask = torch.zeros([xs_pad.size(0), self.hop_size, xs_pad.size(2)]).to(xs_pad.device)        
        xs_pad_r = torch.cat([xs_pad, mask], dim=1)
        xs_pad_r = xs_pad_r.narrow(1, self.hop_size, xs_pad.size(1))
        
        if is_3steps:
            xs_pad_rr = torch.cat([xs_pad_r, mask], dim=1)
            xs_pad_rr = xs_pad_rr.narrow(1, self.hop_size, xs_pad.size(1))
        else:
            xs_pad_rr = None
                    
        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        assert self.block_size > 0
#         print("block_num:", block_num, xs_pad.shape)
        if n_processed_blocks == 0 and total_frame_num <= self.block_size and is_final:
            xs_chunk = self.pos_enc(xs_pad).unsqueeze(1)
            xs_pad, _, _, _, _, _, _ = self.encoders(
                xs_chunk, None, True, None, None, True
            )
            xs_pad = xs_pad.squeeze(0)
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)
                
#             print("too short:", xs_pad.shape)
            return (xs_pad, xs_pad, xs_pad), None, None

        # start block processing
        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        )
        
        xs_chunk_r = xs_pad_r.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad_r.size(-1)
        )
        
        if is_3steps:
            xs_chunk_rr = xs_pad_rr.new_zeros(
                bsize, block_num, self.block_size + 2, xs_pad_rr.size(-1)
            )                        
                       
        for i in range(block_num):
            cur_hop = i * self.hop_size
            chunk_length = min(self.block_size, total_frame_num - cur_hop)
            
            addin = xs_pad.narrow(1, cur_hop, chunk_length)
            addin_r = xs_pad_r.narrow(1, cur_hop, chunk_length)
            if is_3steps:
                addin_rr = xs_pad_rr.narrow(1, cur_hop, chunk_length)
            if self.init_average:
                addin = addin.mean(1, keepdim=True)
                addin_r = addin_r.mean(1, keepdim=True)
                if is_3steps:
                    addin_rr = addin_rr.mean(1, keepdim=True)
            else:
                addin = addin.max(1, keepdim=True)
                addin_r = addin_r.max(1, keepdim=True)
                if is_3steps:
                    addin_rr = addin_rr.max(1, keepdim=True)
                
            if self.ctx_pos_enc:
                addin = self.pos_enc(addin, i + n_processed_blocks)
                addin_r = self.pos_enc(addin_r, i + n_processed_blocks)
                if is_3steps:
                    addin_rr = self.pos_enc(addin_rr, i + n_processed_blocks)

            if prev_addin is None:
                prev_addin = addin
              
            # Prev addin for Parallel Encoding
            if prev_block is not None:
                prev_block_length = min(prev_block.size(1), self.block_size) - self.hop_size
                prev_addin_r = prev_block.narrow(1, prev_block.size(1) - prev_block_length, prev_block_length)
                prev_addin_r = torch.cat([prev_addin_r, xs_pad.narrow(1, 0, self.hop_size)], dim=1)
                if is_3steps:
                    prev_block_length = min(prev_block.size(1), self.block_size) - self.hop_size * 2
                    prev_addin_rr = prev_block.narrow(1, prev_block.size(1) - prev_block_length, prev_block_length)
                    prev_addin_rr = torch.cat([prev_addin_rr, xs_pad.narrow(1, 0, self.hop_size * 2)], dim=1)
                    
                if self.init_average:                    
                    prev_addin_r = prev_addin_r.mean(1, keepdim=True)
                    if is_3steps:
                        prev_addin_rr = prev_addin_rr.mean(1, keepdim=True)
                else:
                    prev_addin_r = prev_addin_r.max(1, keepdim=True)
                    if is_3steps:
                        prev_addin_rr = prev_addin_rr.max(1, keepdim=True)

                if self.ctx_pos_enc:
                    prev_addin_r = self.pos_enc(prev_addin_r, max(i + n_processed_blocks - 1, 0))
                    if is_3steps:
                        prev_addin_rr = self.pos_enc(prev_addin_rr, max(i + n_processed_blocks - 1, 0))
            else:
                prev_addin_r = addin_r                
                if is_3steps:
                    prev_addin_rr = addin_rr
                
            xs_chunk[:, i, 0] = prev_addin
            xs_chunk[:, i, -1] = addin
            
            xs_chunk_r[:, i, 0] = prev_addin_r
            xs_chunk_r[:, i, -1] = addin_r
            
            if is_3steps:
                xs_chunk_rr[:, i, 0] = prev_addin_rr
                xs_chunk_rr[:, i, -1] = addin_rr

            chunk = self.pos_enc(
                xs_pad.narrow(1, cur_hop, chunk_length),
                cur_hop + self.hop_size * n_processed_blocks,
            )
            
            chunk_r = self.pos_enc(
                xs_pad_r.narrow(1, cur_hop, chunk_length),
                cur_hop + self.hop_size * n_processed_blocks,# + self.look_ahead, ## ここダメ
            )
                        

            xs_chunk[:, i, 1 : chunk_length + 1] = chunk
            xs_chunk_r[:, i, 1 : chunk_length + 1] = chunk_r

            prev_addin = addin
            prev_block = xs_pad.narrow(1, cur_hop, chunk_length)
            
            if is_3steps:
                chunk_rr = self.pos_enc(
                    xs_pad_rr.narrow(1, cur_hop, chunk_length),
                    cur_hop + self.hop_size * n_processed_blocks, # + self.look_ahead,
                )
                xs_chunk_rr[:, i, 1 : chunk_length + 1] = chunk_rr

        # mask setup, it should be the same to that of forward_train
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
        mask_online.narrow(2, 1, self.block_size + 1).narrow(
            3, 0, self.block_size + 1
        ).fill_(1)      
        
        if not is_3steps:
            mask_online_r = xs_pad_r.new_zeros(
                xs_pad_r.size(0), block_num, self.block_size + 2, self.block_size + 2
            )
            mask_online_r.narrow(2, 1, self.block_size + 1).narrow(
                3, 0, self.block_size + 1 - self.look_ahead   # masking look-ahead frames
            ).fill_(1)
        else:
            mask_online_r = xs_pad_r.new_zeros(
                xs_pad_r.size(0), block_num, self.block_size + 2, self.block_size + 2
            )
            mask_online_r.narrow(2, 1, self.block_size + 1).narrow(
                3, 0, self.block_size + 1 - (self.look_ahead // 2)   # masking look-ahead frames
            ).fill_(1)
            mask_online_rr = xs_pad_rr.new_zeros(
                xs_pad_rr.size(0), block_num, self.block_size + 2, self.block_size + 2
            )
            mask_online_rr.narrow(2, 1, self.block_size + 1).narrow(
                3, 0, self.block_size + 1 - self.look_ahead   # masking look-ahead frames
            ).fill_(1)
                    
        ys_chunk, _, _, _, past_encoder_ctx, _, _ = self.encoders(
            xs_chunk, mask_online, True, past_encoder_ctx
        )
        
        ys_chunk_r, _, _, _, past_encoder_ctx_r, _, _ = self.encoders(
            xs_chunk_r, mask_online_r, True, past_encoder_ctx_r
        )        
        
        if is_3steps:
            ys_chunk_rr, _, _, _, past_encoder_ctx_rr, _, _ = self.encoders(
                xs_chunk_rr, mask_online_rr, True, past_encoder_ctx_rr
            )        
        
        # remove addin                
        ys_chunk = ys_chunk.narrow(2, 1, self.block_size)
        ys_chunk_r = ys_chunk_r.narrow(2, 1, self.block_size)        
        if is_3steps:
            ys_chunk_rr = ys_chunk_rr.narrow(2, 1, self.block_size)
        
#         print("ys_chunk:", ys_chunk.shape)
        offset = self.block_size - self.look_ahead - self.hop_size        
        if is_final:
            #y_r_length = 0
            if n_processed_blocks == 0:
                y_length = xs_pad.size(1)
                y_r_length = xs_pad_r.size(1)
                if is_3steps:
                    y_rr_length = xs_pad_rr.size(1)
            else:
#                 y_length = xs_pad.size(1) - offset - 
#                 y_r_length = xs_pad_r.size(1) - offset
#                 if is_3steps:
#                     y_rr_length = xs_pad_rr.size(1) - offset
                y_length = block_num * self.hop_size
                y_r_length = block_num * self.hop_size
                if is_3steps:
                    y_rr_length = block_num * self.hop_size                    
    
#             print(y_length, y_r_length, y_rr_length)
        else:
            y_length = block_num * self.hop_size
            y_r_length = block_num * self.hop_size
            if is_3steps:
                y_rr_length = block_num * self.hop_size
            if n_processed_blocks == 0:
                y_length += offset
                y_r_length += offset
                if is_3steps:
                    y_rr_length += offset
                
        ys_pad = xs_pad.new_zeros((xs_pad.size(0), y_length, xs_pad.size(2)))
        ys_pad_r = xs_pad_r.new_zeros((xs_pad_r.size(0), y_r_length, xs_pad_r.size(2)))
        if is_3steps:
            ys_pad_rr = xs_pad_rr.new_zeros((xs_pad_rr.size(0), y_rr_length, xs_pad_rr.size(2)))
        else:
            ys_pad_rr = None
            
        if n_processed_blocks == 0:
            ys_pad[:, 0 : offset] = ys_chunk[:, 0, 0 : offset]
            ys_pad_r[:, 0 : offset] = ys_chunk_r[:, 0, 0 : offset]
            if is_3steps:
                ys_pad_rr[:, 0 : offset] = ys_chunk_rr[:, 0, 0 : offset]
        for i in range(block_num):
            cur_hop = i * self.hop_size
            if n_processed_blocks == 0:
                cur_hop += offset
            if i == block_num - 1 and is_final:
                chunk_length = min(self.block_size - offset, ys_pad.size(1) - cur_hop)                
            else:
                chunk_length = self.hop_size
                
            ys_pad[:, cur_hop : cur_hop + chunk_length] = ys_chunk[
                :, i, offset : offset + chunk_length
            ]
            
            ys_pad_r[:, cur_hop  : cur_hop + chunk_length] = ys_chunk_r[
                    :, i, offset : offset + chunk_length
                ]
                
            if is_3steps:    
                ys_pad_rr[:, cur_hop  : cur_hop + chunk_length] = ys_chunk_rr[
                    :, i, offset : offset + chunk_length
                ]
                
        if n_processed_blocks == 0:
            ys_pad_r = ys_pad_r.narrow(1, offset, ys_pad_r.size(1)-offset)
            if is_3steps:    
                ys_pad_rr = ys_pad_rr.narrow(1, offset, ys_pad_rr.size(1)-offset)
                
        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)
            ys_pad_r = self.after_norm(ys_pad_r)
            if is_3steps:    
                ys_pad_rr = self.after_norm(ys_pad_rr)

        if is_final:
            next_states = None
        else:
            next_states = {
                "prev_addin": prev_addin,
                "prev_block": prev_block,
                "buffer_before_downsampling": buffer_before_downsampling,
                "ilens_buffer": ilens_buffer,
                "buffer_after_downsampling": buffer_after_downsampling,
                "n_processed_blocks": n_processed_blocks + block_num,
                "past_encoder_ctx": past_encoder_ctx,
                "past_encoder_ctx_r": past_encoder_ctx_r,
                "past_encoder_ctx_rr": past_encoder_ctx_rr,
            }
        
#         print("ys_pad:", ys_pad.shape, ys_pad_r.shape, ys_pad_rr.shape)
        return (ys_pad, ys_pad_r, ys_pad_rr), None, next_states
