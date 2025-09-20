import torch
import fakequant    # 假量化替换代码
from datasets import load_dataset
from evaluate import Evaluator
from newLoader import LlmModelLoader  # ✅ 你刚写的模型加载类


def run_inference(model, tokenizer, prompt, max_new_tokens=50):
    """给定模型 + tokenizer + prompt，运行推理并返回文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    model.config.use_cache = True  # 强制启用缓存
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 不采样，保证可重复性
            use_cache=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # === 1️⃣ 选择模型路径 ===
    model_name = "/root/autodl-tmp/model/Qwen2.5-1.5B-Instruct"
    access_token = None  # 如果是 gated 模型（比如 LLaMA3）就填 token

    # === 2️⃣ 使用自定义加载器加载模型 ===
    print(f"🔄 Loading original model {model_name} ...")
    loader = LlmModelLoader(
        model_name_or_path=model_name,
        access_token=access_token,
        dtype="float32",
        max_length=2048
    )
    model = loader.get_model().cuda()
    tokenizer = loader.get_tokenizer()

    # === 3️⃣ 定义 prompt ===
    prompt = "Please tell me a story."

    # === 4️⃣ 原始模型推理 ===
    print("🚀 Running ORIGINAL model inference...")
    output_orig = run_inference(model, tokenizer, prompt)
    print("\n=== ✅ Original Model Output ===")
    print(output_orig)
    
    
    # === 5️⃣ 测试困惑度 ===
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    evaluator = Evaluator(dataset, tokenizer, device="cuda",
                          n_samples=20, ctx_len=2048, stride=2048)
    ppl_fp16 = evaluator.evaluate(model)
    print(f"Original model (fp16) perplexity: {ppl_fp16}")
    
    # === 6️⃣ 替换为假量化线性层（FakeQuantLinear） ===
    print("\n🔄 Applying FakeQuant W8A8 (per_channel weight, per_token activation)...")
    model_q = fakequant.quantize_qwen(
        model,
        weight_quant="per_channel",  # 权重量化方式
        act_quant="per_token",       # 激活量化方式
        w_bits=8,                    # 权重量化 
        a_bits=8,                    # 激活量化 
        kv_bits=8,                   # kv-cache量化
        quantize_kv=True             # 是否kv-cache量化
    ).cuda()
 
    # === 7️⃣ 量化后模型推理 ===
    print("🚀 Running QUANTIZED model inference (W8A8)...")
    output_quant = run_inference(model_q, tokenizer, prompt)
    print("\n=== ✅ Quantized Model Output ===")
    print(output_quant)
    
    # === 8️⃣ 量化后困惑度 ===
    ppl_quant = evaluator.evaluate(model_q)
    print(f"Quantized model perplexity: {ppl_quant}")
    

if __name__ == "__main__":
    main()
