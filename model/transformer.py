from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

def generate_original_PE(length: int, d_model: int) -> torch.Tensor:

    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:

    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:

    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)

        # Compute local map mask
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self._attention_size, mask_future=False, device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 chunk_size: Optional[int] = 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._chunk_size = chunk_size

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, self._chunk_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        K = query.shape[1]
        n_chunk = K // self._chunk_size

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._chunk_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(
            n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention


class MultiHeadAttentionWindow(MultiHeadAttention):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 window_size: Optional[int] = 168,
                 padding: Optional[int] = 168 // 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._window_size = window_size
        self._padding = padding
        self._q = q
        self._v = v

        # Step size for the moving window
        self._step = self._window_size - 2 * self._padding

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, self._window_size)), diagonal=1).bool(),
                                         requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._window_size, self._attention_size),
                                                requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:

        batch_size = query.shape[0]

        # Apply padding to input sequence
        query = F.pad(query.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        key = F.pad(key.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)
        value = F.pad(value.transpose(1, 2), (self._padding, self._padding), 'replicate').transpose(1, 2)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Divide Q, K and V using a moving window
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape((-1, self._v, self._window_size)).transpose(1, 2)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._window_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Fold chunks back
        attention = attention.reshape((batch_size*self._h, -1, self._window_size, self._v))
        attention = attention[:, :, self._padding:-self._padding, :]
        attention = attention.reshape((batch_size*self._h, -1, self._v))

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.
    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).
    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.
        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.
        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.relu(self._linear1(x)))

class Encoder(nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        """Initialize the Encoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.
        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.
        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map


class Transformer(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.2,
                 chunk_mode: str = 'chunk',
                 pe: str = None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        self.softmax = nn.Softmax(dim=d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = x.shape[1]

        # Embeddin module
        encoding = x
        # encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)

        # Output module
        output = self._linear(encoding)
        output = torch.sigmoid(output)
        return output

class CNN(nn.Module):
    def __init__(self, in_channels=12, num_classes=108):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=14, padding=2, stride=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=14, padding=0,stride=3)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=10, padding=0,stride = 2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=10, padding=0,stride=2)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=10, padding=0,stride=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=10, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

import torch
if __name__ == '__main__':
    x = torch.randn(1, 12, 3000)
    c = CNN()
    m = Transformer()
    flops, params = get_model_complexity_info(m, (12, 18000), as_strings=True, print_per_layer_stat=True)
    print("%s |%s" % (flops, params))
    print(c)
    y = c(x)
    y = m(x)
    print('done')