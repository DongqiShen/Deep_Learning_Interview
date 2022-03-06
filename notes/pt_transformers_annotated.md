# pytorch Transformer
---
源码分析transformer的pytorch实现，包括encoder和decoder，以及多头注意力机制。
## class TransformerEncoder
TransformeEncoder代表了整个transformer的encoder，是N个transformerEncoderLayer的叠加结构。
### Args 初始化参数
  - **encoder_layer**：单个TransformerEncoderLayer
  - **num_layers**：叠加的TransformerEncoderLayer的个数
  - norm：正则化的方法
### def __init__(self, ...) 初始化函数
参数如上所示
```python
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        # 由于传入的只是一层transformerencoderlayer，这里直接进行拷贝
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
```
### def forward(self, ...)
输入后进行计算
**参数说明**
  - **src**：单词词向量的序列
  - mask：输入序列的mask
  - src_key_padding_mask：每个batch中src keys的mask.
```python
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        # 上一层output作为下一层的输入
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output
```

## class TransformerEncoderLayer
TransformerEncoderLayer由self-attn（自注意力）和feedforward network（全连接网络）这两个模块组成
### Args 初始化参数
  - **d_model**：特征维度，即一个词用d_model维度的向量表示。
  - **n_head**：注意力机制中多头的数量，用于构造multiheadattention模块。
  - dim_feedforward：全连接层节点数量。
  - dropout：dropout的比例，为[0, 1]之间的浮点数。
  - activation：中间层的激活函数，默认为"relu"。
  - layer_norm_eps：为了防止正则化中的分母为0而设置的一个值，默认为1e-5。
  - batch_first："true"：输入输出的维度则为（batch, seq, feature）。
  - norm_first："true"：正则化层放在attention层和feedforward之前。

### def __init__(...) 初始化函数
参数如上所示
```python
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        # 多头注意力层
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        # linear1和linear2的作用是缩放，首先linear1将向量映射到一个更大的空间
        # 其次linear2再映射回向量原来的大小
        # 他们都属于feedfarward层
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        # norm1和norm2分别作用于encoderlayer中的两个模块，attn和feedforward。
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
```

### def forward(self, ...)
输入的正向计算过程.
**参数说明**:
  - **src**: 输入的词向量序列。
  - src_mask：序列的mask，用于在计算attn时选择用到的词。
  - src_key_padding_mask：每个batch中src keys的mask。
```python
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        # 这里运用到了resnet，把输入加在输出上面
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    # 这里最终的输出没有经过激活函数，为原始的logits
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
```

## class TransformerDecoder
TransformerDecoder为Transformer的decoder，由多个TransformerDecoderLayer叠加起来构成。
### Args 初始化参数
  - **decoder_layer**：一个TransformerDecoderLayer模块
  - **num_layers** ：decoder_layer叠加的个数，在该函数内部进行复制
  - norm：正则化层
### def __init__(self, ...)
参数如上所示
```python
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
```
### def forward(self, ...)
正向计算过程
**参数说明**
  - **tgt**： 词向量的输入序列
  - **memory**：encoder的输出，作为decode中第二层attn的key和value
  - tgt_mask：tgt序列的mask，用于遮盖该单词之后的序列，不进入计算attn
  - memory_mask：memory序列的mask。
  - tgt_key_padding_mask：每个batch中的tgt key的mask。[TODO]
  - memory_key_padding_mask：每个batch中的memory key的mask。[TODO]
```python
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        # 上一层decoder layer的输出（output）作为下一层的输入
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output
```


## class TransformerDecoderLayer
表示一个transformer decoder layer，每个layer由三个子模块组成，self-attn，multi-head-attn和feedforward。初始化参数和encoder一样。相比于encoder，多了一个multi-head-attn，用到了mask，使得这个attn层只能按照顺序获得输入，并且当前输入只能看到之前的输入。
### Args 初始化参数
  - **d_model**：特征的维度
  - **nhead**：多头的数量，用于初始化multiheadattention模块
  - dim_feedforward：全连接层的节点数
  - dropout：dropout的值
  - activation：中间层的激活函数
  - layer_norm_eps：正则化中分母的最小值，防止除0
  - batch_first：是否将batch的维度放在第0的位置
  - norm_first：是否在attn和feedforward之前做正则化。
### def __init__(self, ...)
参数如上
```python
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # ！！这是不同于encoder的地方，当前的输入只能看到之前的输入
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
```
### def forward(self, ...)
输入后模型的计算过程
**参数说明**
  - **tgt**：输入的序列
  - **memory**：encoder的输出，作为decoder中attn的key和value。
  - tgt_mask：decoder输入的mask，屏蔽改单词之后的序列
  - memory_mask：encoder输出的mask。[TODO：不确定用法]
  - tgt_key_padding_mask：每个batch的key的mask。[TODO]
  - memory_key_padding_mask：每个batch中memory的key的mask。[TODO]
```python
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        # 同样用到了残差连接，在论文中的实现中，norm层放在最后（else分支）
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    # 最底层的attn层，用来处理decoder的输入
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    # 中间的attn层，x为底层attn的输出作为query，mem为encoder的输出，作为keys和value
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    # 位于整个decoderlayer的最上层，和encoder一样
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
```

## class Transformer:
完整的Transformer模型，包含了encoder和decoder两部分。
### Args 初始化参数
  - **d_model**：特征长度，即词向量的维度。
  - **nhead**：多头的数量。
  - num_encoder_layers：叠加encoder layer的个数
  - num_decoder_layers：叠加decoder layer的个数
  - dim_feedforward：全连接层的节点个数
  - dropout：dropout的值
  - activation：中间层的激活函数
  - custom_encoder：自定义的encoder。
  - custom_decoder：自定义的decoder。
  - layer_norm_eps：正则化中分母的最小值，防止除以0。
  - batch_first：是否把batch的维度放在第0维
  - norm_first：是否把正则化放在attn和feedforward之前
### def __init__(self, ...)
参数如上
```python
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            # 构造一个encoder layer
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            # 构造完整的encoder
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            # 构造一个decoder layer
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            # 构造完整的decoder
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
```

### def forward(self, ...)
前向计算过程
**参数说明**
  - **src**：输入encoder的词向量序列。
  - **tgt**：输入decoder的词向量序列。
  - src_mask：encoder的mask。
  - tgt_mask：decoder的mask。
  - memory_mask：encoder输出的mask。
  - src_key_padding_mask：每个batch的src keys的mask。
  - tgt_key_padding_mask：每个batch的tgt keys的mask。
  - memory_key_padding_mask：每个batch的memory keys的mask。

**Shape说明**: S: source 序列长度，T：target 序列长度，N：batch大小，E：特征的维度
  - src: (S, N, E), if batch_first then(N, S, E).
  - tgt: (T, N, E), if batch_first then(N, T, E).
  - src_mask: (S, S).
  - tgt_mask: (T, T).
  - memory_mask: (T, S).
  - src_key_padding_mask: (N, S).
  - tgt_key_padding_mask: (N, T).
  - memory_key_padding_mask: (N, S).

  - output: (T, N, E), if batch_first then (N, T, E)
```python
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask:    Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        # 分为encoder和decoder，memory是encoder的输入，作为decoder的一个输入
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output
```

## class MultiheadAttention
多头注意力模型，其注意力公式如下:
$$
Multihead(Q, K, V) = Concat(head_1, \dots, head_h) * W^O \\
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
### Args 初始化参数
  - embed_dim: 模型总的维度
  - num_heads: 并行的多头的数量。embed_dim会被num_heads进行分割，每一个头的维度为 embeded_dim // num_heads。
  - dropout：dropout的值。
  - bias：对输入和输入的映射层添加bias。默认为添加。
  - add_bias_kv：对key和value在维度0添加bias。
  - add_zero_attn：对keys和value序列在维度1添加一个新的为0的batch。
  - kdim：keys的维度。默认为kdim=embed_dim。
  - vdim：values的维度。默认为vdim=embed_dim。
  - batch_first：true: (batch, seq, feature), false: (seq, batch, feature)。

### def __init__(self, ...)
初始化参数如上
```python
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # 如果key和value的维度不一样，就需要加一层全连接层，映射到同样的大小。
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()


    def _reset_parameters(self):
        """参数初始化函数"""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)
```

### def forward(self, ...)
输入后模型的计算过程

**参数说明**
  - **query**： 形状为(L, N, E_q)，L为目标序列的最大长度，N为batch大小，E_q为特征的维度。
  - **key**：形状为(S, N, E_q)，S为序列长度，N为batch大小，E_q为特征的维度。
  - **value**：形状为(S, N, E_q)，S为序列长度，N为batch大小，E_q为特征的维度。
  - key_padding_mask：形状为(N, S)，N为batch大小，S为序列长度。表示key序列中的某些key不想要参与计算attention。
  - need_weights：if true，输出attn的值
  - attn_mask：形状为(L, S) 或 (N * num_heads, L, S)。其中N为batch大小，L为目标序列的长度，S为输入序列的长度。

```python
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
```