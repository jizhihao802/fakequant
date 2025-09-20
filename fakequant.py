import torch
import torch.nn as nn
from functools import partial

# 此fakequant 原地修改了传入的模型model,入需要保留原model可使用copy.deepcopy(model)
# =============== 基础量化函数 ===============

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    """ 权重 per-channel 量化（按输出维度）"""
    if n_bits >= 16:
        return w  # 不量化
    q_max = 2 ** (n_bits - 1) - 1
    scales = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6) / q_max
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    """ 权重 per-tensor 量化（全局一个 scale）"""
    if n_bits >= 16:
        return w
    q_max = 2 ** (n_bits - 1) - 1
    scale = w.abs().max().clamp(min=1e-6) / q_max
    w.div_(scale).round_().mul_(scale)
    return w

@torch.no_grad()
def quantize_activation_per_token_absmax(x, n_bits=8):
    """ 激活 per-token 量化 [B, Seq, H] -> 每个 token 一个 scale """
    if n_bits >= 16:
        return x  # 不量化
    q_max = 2 ** (n_bits - 1) - 1
    scales = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6) / q_max
    x.div_(scales).round_().mul_(scales)
    return x

@torch.no_grad()
def quantize_activation_per_tensor_absmax(x, n_bits=8):
    """ 激活 per-tensor 量化 [B, Seq, H] -> 全局一个 scale """
    if n_bits >= 16:
        return x
    q_max = 2 ** (n_bits - 1) - 1
    scale = x.abs().max().clamp(min=1e-6) / q_max
    x.div_(scale).round_().mul_(scale)
    return x


# =============== FakeQuantLinear 支持 W_bits / A_bits ===============

class FakeQuantLinear(nn.Module):
    """
    Fake Quant Linear for LLaMA/Qwen
    - w_bits: 权重量化 bit (静态量化)
    - a_bits: 激活量化 bit (forward 动态量化)
    - 支持：
        W 32/16/8/4
        A 32/16/8/4
        KV 32/16/8/4
    """
    def __init__(self, in_features, out_features, bias=True,
                 weight_quant="per_channel",
                 act_quant="per_token",
                 w_bits=8,
                 a_bits=8,
                 kv_bits=32,
                 quantize_kv=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.kv_bits = kv_bits
        self.quantize_kv = quantize_kv
        self.weight_quant_name = weight_quant
        self.act_quant_name = act_quant
        # ======= 激活量化函数 =======
        if act_quant == "per_token":
            self.act_quant_fn = quantize_activation_per_token_absmax
        elif act_quant == "per_tensor":
            self.act_quant_fn = quantize_activation_per_tensor_absmax
        else:
            raise ValueError(f"Invalid act_quant={act_quant}")

        # ======= 初始化存储量化后权重 =======
        self.register_buffer("quantized_weight", None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        # 1️⃣ 激活量化 (动态)
        x_q = self.act_quant_fn(x, self.a_bits)
        # 2️⃣ 直接使用静态量化后的权重
        y = torch.nn.functional.linear(x_q, self.quantized_weight, self.bias)
        # 3️⃣ 是否有kv-cache量化
        if self.quantize_kv:
            y = self.act_quant_fn(y, self.kv_bits)
        return y

    @staticmethod
    def from_float(module: nn.Linear,
                   weight_quant="per_channel",
                   act_quant="per_token",
                   w_bits=8,
                   a_bits=8,
                   kv_bits=32,
                   quantize_kv=False):
        """从 nn.Linear 复制参数并**静态量化权重**"""
        new_module = FakeQuantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            weight_quant=weight_quant,
            act_quant=act_quant,
            w_bits=w_bits,
            a_bits=a_bits,
            kv_bits=kv_bits,
            quantize_kv=quantize_kv
        )

        # ==== 先取出原始权重 ====
        orig_dtype = module.weight.dtype  # 记录原始数据类型
        orig_w = module.weight.data.clone().to(orig_dtype)

        # ==== 静态量化权重 ====
        if w_bits >= 16:
            w_q = orig_w  # 不量化
        elif weight_quant == "per_channel":
            w_q = quantize_weight_per_channel_absmax(orig_w, n_bits=w_bits)
        elif weight_quant == "per_tensor":
            w_q = quantize_weight_per_tensor_absmax(orig_w, n_bits=w_bits)
        else:
            raise ValueError(f"Invalid weight_quant={weight_quant}")

        # ==== 保存静态量化后的权重 ====
        new_module.register_buffer("quantized_weight", w_q.to(orig_dtype))

        # ==== bias 原样拷贝 ====
        if module.bias is not None:
           new_module.bias.data.copy_(module.bias.data.to(orig_dtype))

        return new_module

    def __repr__(self):
        return (f"FakeQuantLinear({self.in_features}, {self.out_features}, "
                f"W{self.w_bits}, A{self.a_bits}, "
                f"weight_q={self.weight_quant_name}, act_q={self.act_quant_name}, "
                f"kv_q={self.quantize_kv}, STATIC_WQ)")


# =============== 替换模型里的 Linear + KV 量化 ===============

def quantize_llama(model,
                   weight_quant="per_channel",
                   act_quant="per_token",
                   w_bits=8,
                   a_bits=8,
                   kv_bits=32,
                   quantize_kv=False):
    """替换 LLaMA 模型中的 Linear 层为 FakeQuantLinear，并处理 KV Cache"""
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    for name, m in model.named_modules():
        if isinstance(m, LlamaAttention):
            m.q_proj = FakeQuantLinear.from_float(
                m.q_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.k_proj = FakeQuantLinear.from_float(
                m.k_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.v_proj = FakeQuantLinear.from_float(
                m.v_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.o_proj = FakeQuantLinear.from_float(
                m.o_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, quantize_kv=False)


        elif isinstance(m, LlamaMLP):
            m.gate_proj = FakeQuantLinear.from_float(
                m.gate_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.up_proj = FakeQuantLinear.from_float(
                m.up_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.down_proj = FakeQuantLinear.from_float(
                m.down_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)

    return model

def quantize_qwen(model,
                  weight_quant="per_channel",
                  act_quant="per_token",
                  w_bits=8,
                  a_bits=8,
                  kv_bits=32,
                  quantize_kv=False):
    """替换 Qwen 模型中的 Linear 层为 FakeQuantLinear，并处理 KV Cache"""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

    for name, m in model.named_modules():
        if isinstance(m, Qwen2Attention):
            m.q_proj = FakeQuantLinear.from_float(
                m.q_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.k_proj = FakeQuantLinear.from_float(
                m.k_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.v_proj = FakeQuantLinear.from_float(
                m.v_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.o_proj = FakeQuantLinear.from_float(
                m.o_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, quantize_kv=False)


        elif isinstance(m, Qwen2MLP):
            m.gate_proj = FakeQuantLinear.from_float(
                m.gate_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.up_proj = FakeQuantLinear.from_float(
                m.up_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.down_proj = FakeQuantLinear.from_float(
                m.down_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)

    return model

def quantize_qwen3(model,
                  weight_quant="per_channel",
                  act_quant="per_token",
                  w_bits=8,
                  a_bits=8,
                  kv_bits=32,
                  quantize_kv=False):
    """替换 Qwen 模型中的 Linear 层为 FakeQuantLinear，并处理 KV Cache"""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

    for name, m in model.named_modules():
        if isinstance(m, Qwen3Attention):
            m.q_proj = FakeQuantLinear.from_float(
                m.q_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.k_proj = FakeQuantLinear.from_float(
                m.k_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.v_proj = FakeQuantLinear.from_float(
                m.v_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.o_proj = FakeQuantLinear.from_float(
                m.o_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits, quantize_kv=False)


        elif isinstance(m, Qwen3MLP):
            m.gate_proj = FakeQuantLinear.from_float(
                m.gate_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.up_proj = FakeQuantLinear.from_float(
                m.up_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)
            m.down_proj = FakeQuantLinear.from_float(
                m.down_proj, weight_quant, act_quant, w_bits=w_bits, a_bits=a_bits)

    return model


def quantize_gemma(model,
                   weight_quant="per_channel",
                   act_quant="per_token",
                   w_bits=8,
                   a_bits=8,
                   kv_bits=32,
                   quantize_kv=False):
    """
    替换 Gemma 模型里的所有 Linear 为 FakeQuantLinear
    同时对 KV Cache 做 fake quant (32/16/8/4)
    """
    from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaMLP

    for name, m in model.named_modules():
        if isinstance(m, GemmaAttention):
            # Attention q/k/v/o
            m.q_proj = FakeQuantLinear.from_float(
                m.q_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                kv_bits=kv_bits, quantize_kv=quantize_kv
            )
            m.k_proj = FakeQuantLinear.from_float(
                m.k_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                kv_bits=kv_bits, quantize_kv=quantize_kv
            )
            m.v_proj = FakeQuantLinear.from_float(
                m.v_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                kv_bits=kv_bits, quantize_kv=quantize_kv
            )
            m.o_proj = FakeQuantLinear.from_float(
                m.o_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                quantize_kv=False
            )

        elif isinstance(m, GemmaMLP):
            # MLP
            m.gate_proj = FakeQuantLinear.from_float(
                m.gate_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits
            )
            m.up_proj = FakeQuantLinear.from_float(
                m.up_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits
            )
            m.down_proj = FakeQuantLinear.from_float(
                m.down_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits
            )

    return model

def quantize_minicpm(model,
                     weight_quant="per_channel",
                     act_quant="per_token",
                     w_bits=8,
                     a_bits=8,
                     kv_bits=32,
                     quantize_kv=False):
    """
    替换 MiniCPM 模型里的 Linear 层为 FakeQuantLinear
    通过类名字符串匹配执行替换（适配动态模块）
    """

    for name, m in model.named_modules():
        cls_name = m.__class__.__name__
        #print(cls_name)
        if cls_name == "MiniCPMSdpaAttention":
            m.q_proj = FakeQuantLinear.from_float(m.q_proj, weight_quant, act_quant,
                                                  w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.k_proj = FakeQuantLinear.from_float(m.k_proj, weight_quant, act_quant,
                                                  w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.v_proj = FakeQuantLinear.from_float(m.v_proj, weight_quant, act_quant,
                                                  w_bits=w_bits, a_bits=a_bits, kv_bits=kv_bits, quantize_kv=quantize_kv)
            m.o_proj = FakeQuantLinear.from_float(m.o_proj, weight_quant, act_quant,
                                                  w_bits=w_bits, a_bits=a_bits, quantize_kv=False)

        elif cls_name == "MiniCPMMLP":
            m.gate_proj = FakeQuantLinear.from_float(m.gate_proj, weight_quant, act_quant,
                                                     w_bits=w_bits, a_bits=a_bits)
            m.up_proj = FakeQuantLinear.from_float(m.up_proj, weight_quant, act_quant,
                                                   w_bits=w_bits, a_bits=a_bits)
            m.down_proj = FakeQuantLinear.from_float(m.down_proj, weight_quant, act_quant,
                                                     w_bits=w_bits, a_bits=a_bits)

    return model

def quantize_phi(model,
                  weight_quant="per_channel",
                  act_quant="per_token",
                  w_bits=8,
                  a_bits=8,
                  kv_bits=32,
                  quantize_kv=False):
    """
    替换 Phi-3 模型中的 Linear 层为 FakeQuantLinear（qkv_proj 和 gate_up_proj 是融合的）
    """

    for name, m in model.named_modules():
        cls_name = m.__class__.__name__
        if cls_name == "Phi3Attention":
            m.qkv_proj = FakeQuantLinear.from_float(
                m.qkv_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                kv_bits=kv_bits, quantize_kv=quantize_kv
            )
            m.o_proj = FakeQuantLinear.from_float(
                m.o_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits,
                quantize_kv=False
            )

        elif cls_name == "Phi3MLP":
            m.gate_up_proj = FakeQuantLinear.from_float(
                m.gate_up_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits
            )
            m.down_proj = FakeQuantLinear.from_float(
                m.down_proj, weight_quant, act_quant,
                w_bits=w_bits, a_bits=a_bits
            )

    return model

