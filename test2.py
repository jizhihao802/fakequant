import torch
import fakequant    # 假量化替换代码
from datasets import load_dataset
from evaluate import Evaluator
from newLoader import LlmModelLoader  # ✅ 模型加载类
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def run_inference(model, tokenizer, prompt, max_new_tokens=512):
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
    model_name = "/root/autodl-tmp/model/qwen3-4b"
    access_token = None  # 如果是 gated 模型（比如 LLaMA3）就填 token

    # === 2️⃣ 使用自定义加载器加载模型 ===
    print(f"🔄 Loading original model {model_name} ...")
    
    loader = LlmModelLoader(
        model_name_or_path=model_name,
        access_token=access_token,
        dtype="float32",
        max_length=4096
    )
    model = loader.get_model().cuda()
    tokenizer = loader.get_tokenizer()
    
    #model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/model/qwen3-4b").cuda()
    #tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/model/qwen3-4b")
    # === 3️⃣ 添加：读取txt文件，拼接prompt ===
    user_prompt = "\n你是一位秘书，阅读以上的会议内容，负责编写规范的会议纪要。\n\n要求：\n1. 全面提炼会议要点，确保不遗漏任何重要细节。\n2. 严格按照列表形式输出会议纪要，确保格式规范。\n\n输出格式：数字序号后跟随的是你提取的会议要点，短横“-”后跟随的是该要点的细节\n1. xxxxx\n\t- yyyyy\n\t- yyyyyyy \n2. xxxxxxx\n...\n"
    with open("SpinQuant/meeting.txt", "r", encoding="utf-8") as f:
        wiki_text = f.read().strip()
    
    # 统计 token 数
    tokens = tokenizer(wiki_text, return_tensors="pt")  # 转成 PyTorch tensor
    #token_count = len(tokens["input_ids"][0])
    #print("Token 数:", token_count)
    
    
    # === 截取一部分 ===
    max_tokens = 3500 - 110  # 你要设定的最大长度，比如模型 max_length
    truncated_ids = tokens["input_ids"][0, :max_tokens]  # 截断
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    long_text = truncated_text + user_prompt
    short_text = "What is large language model?"
    
    # === 4️⃣ 原始模型推理 ===
    print("🚀 Running ORIGINAL model inference...")
    output_orig = run_inference(model, tokenizer, long_text)
    print("\n=== ✅ Original Model Output ===")
    print(output_orig)
    
    # === 6️⃣ 替换为假量化线性层（FakeQuantLinear） ===
    print("\n🔄 Applying FakeQuant W8A8 (per_channel weight, per_token activation)...")
    model_q = fakequant.quantize_qwen3(
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
    output_quant = run_inference(model_q, tokenizer, long_text)
    print("\n=== ✅ Quantized Model Output ===")
    print(output_quant)
    
if __name__ == "__main__":
    main()
