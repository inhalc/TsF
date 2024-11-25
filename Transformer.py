from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
import copy
import torch

class MyMutiheadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0,bias=Ture):
        super(MyMultiheadAttentiopn,self).__init__()
        """
        embed_dim: 输入的维度
        num_heads: 多头的数量
        dropout: 丢弃率
        bias: 是否使用偏置
        """
        self.embed_dim = embed_dim              # d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim指的就是d_k，d_v的维度
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads              # 多头的数量
        self.dropout = dropout
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须被num_heads整除"
        #论文中d_k = d_v = d_model / num_heads的限制条件
        self.q_proj_weight = Paramoter(torch.Tensor(embed_dim,embed_dim))
        #embed_dim = kdim * num_heads
        #初始化num_heads个W_q堆叠起来，也就是num_heads个头组成embed_dim维度
        self.k_proj_weight = Paramoter(torch.Tensor(embed_dim,embed_dim))
        self.v_proj_weight = Paramoter(torch.Tensor(embed_dim,embed_dim))
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=bias)

    def forward(self,query,key,value,atten_mask = None,key_padding_mask = None):
        """
        编码时 query,key,value都是同一个输入，
        解码时 输入部分也是同一个输入
        解码和编码交互时 key,value指的是memory,query指的是tgt
        quert:              [tgt_len, batch_size, embed_dim], tgt_len为目标序列的长度
        key:                [src_len, batch_size, embed_dim], src_len为源序列的长度
        value:              [src_len, batch_size, embed_dim] 
        atten_mask:         [tgt_len, src_len] or [num_heads*batch_size, tgt_len,src_len], 解码时使用，且一次喂入全部解码部分的输入，用于屏蔽未来信息
        key_padding_mask:   [batch_size, src_len] 用于屏蔽padding的信息
        returun:
        attn_output:        [tgt_len, batch_size, embed_dim]
        attn_output_weight: [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query,key,value,self.num_heads,
                                            self.dropout,self.out_proj.weight,
                                            self.out_proj.bias, training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)
    
    def multi_head_attention_forward(
        query,                  #[tgt_len, batch_size, embed_dim]
        key,                    #[src_len, batch_size, embed_dim]
        value,                  #[src_len, batch_size, embed_dim]
        num_heads,
        dropout_p,
        out_proj_weight,        #[embed_dim = vdim * num_heads, embed_dim]
        out_proj_bias,
        training=True,
        key_padding_mask=None,  #[batch_size, src_len/tgt_len]
        q_proj_weight=None,     
        k_proj_weight=None,
        v_proj_weight=None,
        attn_mask=None,
        ):