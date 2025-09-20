import os
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

class LlmModelLoader:
    def __init__(self, model_name_or_path: str,
                 access_token: str = None,
                 dtype: str = "bfloat16",
                 max_length: int = 2048,
                 tokenizer_path: str = None,
                 lora_path: str = None,
                 trust_remote_code: bool = True):
        self.model_path = model_name_or_path
        self.tokenizer_path = tokenizer_path or model_name_or_path
        self.token = access_token
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "float32":
            self.dtype = torch.float32
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.lora_path = lora_path

        self.model = None
        self.tokenizer = None
        self.config = None
        self.load()

    def load(self):
        model_lower = self.model_path.lower()
        
        # 1. 加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, use_fast=False)
        except:
            self.tokenizer = None
        if None == self.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            except:
                self.tokenizer = None
        if None == self.tokenizer:
            print("Default load tokenizer failed for ", self.model_path)

        # 2. 加载模型
        try:
            if "gemma-3-4b-it" in model_lower:
                from transformers import Gemma3ForConditionalGeneration
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype, token=self.token).eval()
            elif 'gemma-3-1b-it' in model_lower:
                from transformers import Gemma3ForCausalLM
                self.model = Gemma3ForCausalLM.from_pretrained(self.model_path, torch_dtype='auto').eval()
            elif "qwen2.5-omni" in model_lower:
                from transformers import Qwen2_5OmniForConditionalGeneration
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            elif "qwen2.5-vl" in model_lower or "qwen2___5-vl" in model_lower:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            elif "qwen2-vl" in model_lower:
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            elif "qwen2-audio" in model_lower:
                from transformers import Qwen2AudioForConditionalGeneration
                audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype)
                self.model = audio_model.language_model
            elif "llama-3.2" in model_lower and "vision" in model_lower:
                from transformers import MllamaForConditionalGeneration
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            elif "llama" in model_lower or "yi" in model_lower:
                from transformers import LlamaForCausalLM
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code).eval()
            elif "internvl" in model_lower:
                self.model = AutoModel.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    trust_remote_code=self.trust_remote_code).eval()
            elif "smolvlm2" in model_lower:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            elif "smolvlm" in model_lower or "smoldocling" in model_lower:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path, torch_dtype=self.dtype).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, torch_dtype=self.dtype,
                    token=self.token, trust_remote_code=self.trust_remote_code).eval()
        except Exception as e:
            print(f"[Model load failed] {e}")
            raise
        self.config = self.model.config
        
        # 3. 合并 LoRA（可选）
        if self.lora_path:
            try:
                from peft import PeftModel
                adapter = PeftModel.from_pretrained(self.model, self.lora_path)
                self.model = adapter.merge_and_unload(progressbar=True)
            except Exception as e:
                print("[LoRA merge failed]", e)


    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
