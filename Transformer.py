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
        q_proj_weight=None,     #[embed_dim, qdim * num_heads]
        k_proj_weight=None,     #[embed_dim, kdim * num_heads]
        v_proj_weight=None,     #[embed_dim, vdim * num_heads]
        attn_mask=None,         #[tgt_len, src_len]
        ):
        #第一阶段：计算Q,K,V
        q = F.linear(query,q_proj_weight)
        # [tgt_len, batch_size, embed_dim] * [embed_dim, embed_dim] 
        # = [tgt_len, batch_size, embed_dim]
        k = F.linear(key,k_proj_weight)
        # [src_len, batch_size, embed_dim] * [embed_dim, embed_dim]
        # = [src_len, batch_size, embed_dim]
        v = F.linear(value,v_proj_weight)
        # [src_len, batch_size, embed_dim] * [embed_dim, embed_dim]
        # = [src_len, batch_size, embed_dim]

        #第二阶段：缩放，attn_mask维度判断
        tgt_len, bsz, embed_dim = query.size() # [tgt_len, batch_size, embed_dim]
        src_len = key.size(0)
        head_dim = embed_dim // num_heads
        q = q * scaling #[query_len,batch_size,kdim*num_heads]

        if attn_mask is not None:
        #[tgt_len,src_len] or [num_heads*batch_size,tgt_len,src_len]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)#[1,tgt_len,src_len]扩充维度
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif atten_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            #attn_mask维度为3D

        #第三阶段：计算得到注意力权重矩阵
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        # [batch_size * num_heads, tgt_len, qdim]
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        # [batch_size * num_heads, src_len, kdim]
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        # [batch_size * num_heads, src_len, vdim]
        attn_output_weights = torch.bmm(q, k.transpose(1,2))
        # [batch_size * num_heads, tgt_len, qdim] * [batch_size * num_heads, src_len, kdim]
        # = [batch_size * num_heads, tgt_len, src_len]  num_heads个QK相乘后的注意力矩阵

        #第四阶段：mask处理
        if attn_mask is not None:
            attn_output_weights += attn_mask
            #[batch_size*num_heads,tgt_len,scr_len]
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # 变成[batch_size, num_heads, src_len]的形状
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),float('-inf'))
            #扩展维度，从[batch_size, src_len]变成[batch_size,1,1,src_len]
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
            # [batch_size*num_heads,tgt_len,scr_len]

        attn_output_weight = F.softmax(attn_output_weights,dim=-1)
        #[batch_size*num_heads,tgt_len,scr_len]
        attn_output_weight = F.dropout(attn_output_weight,p=dropout_p,training=training)
        attn_output = torch.bmm(attn_output_weight,v)
        #Z = [batch_size*num_heads,tgt_len,src_len] @ [batch_size*num_heads,src_len,vdim] = [batch_size*num_heads,tgt_len,vdim]
        #num_heads个Attention(Q,K,V)结果
        attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len,bsz,embed_dim)
        #先transpose成[tgt_len,bath_size*num_heads,vdim],再view成[tgt_len,batch_size,embed_dim]

        Z = F.linear(attn_output,out_proj_weight,out_proj_bias)
        #这里就是多个 z 线性组合成 Z [tgt_len,batch_size,embed_dim]
        return Z,attn_output_weight.sum(dim=1)/num_heads
        #奖num_heads个注意力权重矩阵按对应维度取平均值