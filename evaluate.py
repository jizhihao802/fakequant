import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset

from transformers import LlamaTokenizer, LlamaForCausalLM

class Evaluator:
    def __init__(self, dataset, tokenizer, device="cuda",
                 n_samples=40, ctx_len=2048, stride=None):
        """
        Args:
            dataset: HF dataset split
            tokenizer: 分词器
            device: "cpu"/"cuda"/"mps"
            n_samples: 总共采 n_samples 个窗口
            ctx_len: 每个窗口的上下文长度（原来是 2048）
            stride: 窗口滑动步幅，默认等于 ctx_len（不重叠）。
                    如果你想要重叠采样，可以设置比如 ctx_len//2
        """
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = n_samples
        self.ctx_len = ctx_len
        self.stride = stride or ctx_len

        # 把整个分好词的大 tensor 一次准备好
        text = "\n\n".join(dataset["text"])
        self.dataset = tokenizer(text, return_tensors="pt").input_ids.to(device)
        self.total_len = self.dataset.size(1)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            start = i * self.stride
            end = start + self.ctx_len
            if end > self.total_len:
                break  # 不够一整块就结束
            batch = self.dataset[:, start:end].to(model.device)

            lm_logits = model(batch).logits
            # batch_length 可能小于 ctx_len（最后一块），要动态获取
            seq_len = batch.size(1)

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]

            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # 按照实际 token 数度量 NLL
            nlls.append(loss * (seq_len - 1))

        # 总 token 数 = sum(seq_len - 1) over windows
        total_tokens = sum((min(self.total_len - i * self.stride, self.ctx_len) - 1)
                           for i in range(len(nlls)))
        return torch.exp(torch.stack(nlls).sum() / total_tokens)

if __name__ == "__main__":
    # 测试代码：Llama 3.2-1B 模型
    local_model_dir = "models/llama-3.2-1B-Instruct"

    # 加载分词器 & 模型（Transformers 会自动识别 .safetensors）
    tokenizer = LlamaTokenizer.from_pretrained(local_model_dir)
    model = LlamaForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},   # 全部放到 CPU
        low_cpu_mem_usage=True,
    )

    # 准备数据 & evaluator
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    evaluator = Evaluator(dataset, tokenizer, device="cpu",
                        n_samples=40, ctx_len=4096, stride=4096)

    # 评测原始 FP16 模型
    ppl_fp16 = evaluator.evaluate(model)
    print(f"Original model (fp16) perplexity: {ppl_fp16}")


