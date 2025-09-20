import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from mpl_toolkits.mplot3d import Axes3D  # 添加在文件顶部

class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def list_modules(self):
        print(self.model.config)
        seen = set()

        def recurse(module, prefix=""):
            for name, child in module.named_children():
                mod_type = type(child)
                if mod_type not in seen:
                    print(f"{prefix}{name}: {child}")
                    seen.add(mod_type)
                    recurse(child, prefix + "  ")
                else:
                    print(f"{prefix}{name}: ({mod_type.__name__} - 已省略重复)")

        recurse(self.model)


    def extract_weights(self, filter_ndim=2, dtype=None, modules=None, module_names=None):
        """
        提取模型中指定模块或全模型符合维度要求的权重，并返回列表（保留原 shape）。
        """
        params = []
        if modules is not None:
            for idx, m in enumerate(modules):
                for pname, p in m.named_parameters():
                    if (isinstance(filter_ndim, int) and p.ndim == filter_ndim) or \
                       (isinstance(filter_ndim, (tuple, list)) and p.ndim in filter_ndim):
                        arr = p.detach()
                        if dtype is not None:
                            arr = arr.to(dtype)
                        params.append(arr.cpu())
        else:
            for pname, p in self.model.named_parameters():
                if (isinstance(filter_ndim, int) and p.ndim == filter_ndim) or \
                   (isinstance(filter_ndim, (tuple, list)) and p.ndim in filter_ndim):
                    arr = p.detach()
                    if dtype is not None:
                        arr = arr.to(dtype)
                    params.append(arr.cpu())
        if not params:
            raise RuntimeError("未找到任何符合条件的权重，请检查 filter_ndim、modules 和 module_names 设置。")
        return params  # 返回权重列表，每个元素保留原 shape

    def register_activation_hooks(self, modules, names=None):
        if names is None:
            names = [f"module_{i}" for i in range(len(modules))]
        self.activations = {n: [] for n in names}

        def make_hook(key):
            def hook(module, inp, out):
                tensor = inp[0] if isinstance(inp, tuple) else inp
                self.activations[key].append(tensor.detach().cpu())  # 不 flatten
            return hook

        for m, name in zip(modules, names):
            h = m.register_forward_hook(make_hook(name))
            self.hooks.append(h)
        self.activation_names = names

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def extract_activations(self, tokenizer, text):
        if not self.hooks:
            raise RuntimeError("请先调用 register_activation_hooks 注册钩子。")
        inputs = tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():       #优化显存
            _ = self.model(**inputs)
        result = {}
        for name in self.activation_names:
            if not self.activations[name]:
                raise RuntimeError(f"模块 {name} 未捕获到任何激活。")
            # 只返回第一个 batch 的激活（通常只有一个）
            result[name] = self.activations[name][0]
        return result

    def plot_distribution(self, tensor, title="Distribution", bins=50, show_extreme=False, extreme_percentile=0.01):
        data = tensor.cpu().numpy()
        vmin, vmax = data.min(), data.max()
        mean, std = data.mean(), data.std()
        print(f"min={vmin}, max={vmax}, mean={mean}, std={std}")

        if data.ndim == 2:
            plt.figure(figsize=(8, 6))
            # 画热力图
            plt.imshow(data, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.title(title + " (heatmap)")
            plt.xlabel('Feature (列号)')
            plt.ylabel('Sample (行号)')

            # 标出极端值位置
            lower = np.percentile(data, extreme_percentile)
            upper = np.percentile(data, 100 - extreme_percentile)
            extreme_mask = (data <= lower) | (data >= upper)
            y_idx, x_idx = np.where(extreme_mask)
            plt.scatter(x_idx, y_idx, c='red', s=8, label='Extreme')
            plt.legend(loc='upper right')

            plt.tight_layout()
            plt.show()
        else:
            plt.figure()
            plt.hist(data.flatten(), bins=bins, range=(vmin, vmax))
            plt.title(title + " (flattened)")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()

    def plot_3d_surface(self, tensor, title="3D Surface", abs_value=True, max_size=300, mode="auto",picture_name='myplot.png'):
        """
        绘制三维绝对值分布图（不降采样）。
        - tensor: 二维权重或激活
        - abs_value: 是否取绝对值
        - max_size: 保留参数但不再使用
        - mode: "weight" or "activation" or "auto"
        """
        data = tensor.cpu().numpy()
        if abs_value:
            data = np.abs(data)
        rows, cols = data.shape
        # 不降采样，直接用全部数据
        data_show = data
        x = np.arange(data_show.shape[1])
        y = np.arange(data_show.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = data_show

        # 自动判断标签
        if mode == "auto":
            if "weight" in title.lower():
                mode = "weight"
            elif "activation" in title.lower():
                mode = "activation"
            else:
                mode = "weight" if rows > 1000 or cols > 1000 else "activation"
        if mode == "weight":
            xlabel, ylabel = "Out Channel", "In Channel"
        else:
            xlabel, ylabel = "Channel", "Token"

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Absolute Value' if abs_value else 'Value')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel('Absolute Value' if abs_value else 'Value')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        plt.savefig(picture_name)  # 保存图片到指定文件

    def inspect_weights(self, filter_ndim=2, dtype=None, modules=None, module_names=None, title="Weights Distribution"):
        ws = self.extract_weights(filter_ndim, dtype, modules, module_names)
        for idx, w in enumerate(ws):
            if w.ndim == 2:
                self.plot_3d_surface(w, title=f"{title} #{idx}", mode="weight",picture_name=f"{module_names[idx]}_weight.png")
            else:
                self.plot_distribution(w, title=f"{title} #{idx}")

    def inspect_activations(self, tokenizer, text, title_map=None):
        acts = self.extract_activations(tokenizer, text)
        for name, tensor in acts.items():
            title = title_map.get(name, f"Activation: {name}") if title_map else f"Activation: {name}"
            # 如果激活是三维（batch, token, channel），取第一个batch
            arr = tensor
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2:
                self.plot_3d_surface(arr, title=title, mode="activation",picture_name=f"{name}_activation.png")
            else:
                self.plot_distribution(arr, title=title)
        self.remove_hooks()

"""
使用示例
from model_inspector import ModelInspector

inspector = ModelInspector(model)

#可以查询模型有哪些层
for name, module in inspector.list_modules():
    print(name, module)

# 可视化指定层权重
# 只可视化 model.layer3 和 model.fc 的权重分布
modules = [model.layer3, model.fc]
names = ['layer3', 'fc']
inspector.inspect_weights(filter_ndim=2, modules=modules, module_names=names, title="Layer3 & FC Weights")

# 可视化某些层的激活
modules = [model.layer1, model.layer2]
names = ['layer1', 'layer2']
给指定模块注册钩子
inspector.register_activation_hooks(modules, names)
inspector.inspect_activations(tokenizer, "示例文本")

"""

# 测试代码：Llama 3.2-3B 模型
if __name__ == "__main__":
    # 加载 Llama 3.2-3B 模型
    local_model_dir = "/root/autodl-tmp/Llama-3.2-3B-Instruct"
    print(f"加载本地模型: {local_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(local_model_dir, device_map="auto", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=False)

    inspector = ModelInspector(model)

    # 列出子模块
    inspector.list_modules()

    # 选择部分线性层进行权重提取
    # 例如，选第一层 self_attn q_proj 和第一层 mlp gate_proj
    layer0 = model.model.layers[0]
    selected_modules = [layer0.self_attn.q_proj, layer0.mlp.gate_proj]
    names = ["layer0_q_proj", "layer0_gate_proj"]

    # 直接调用 inspect_weights 进行可视化
    inspector.inspect_weights(filter_ndim=2, modules=selected_modules, module_names=names, title="Llama3.2-1B Selected Weights")

    # 捕获激活
    inspector.register_activation_hooks(selected_modules, names)
    sample_text = "The quick brown fox jumps over the lazy dog."
    activations = inspector.extract_activations(tokenizer, sample_text)
    for name, act in activations.items():
        print(f"模块 {name} 激活向量长度: {act.numel()}")
    inspector.inspect_activations(tokenizer, sample_text, title_map={n: f"Activation {n}" for n in names})

    print("测试完成。")
