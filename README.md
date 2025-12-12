# 🎨 SDXL Model Merger

<div align="center">

**强大的 Stable Diffusion 模型融合工具**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*通过智能融合多个 Stable Diffusion 模型，创造独一无二的 AI 艺术风格*

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [配置说明](#-配置说明) • [示例](#-示例)

</div>

---

## 🌟 项目简介

SDXL Model Merger 是一个专业的 Stable Diffusion 模型融合工具，支持融合多个 SD1.5/SDXL 检查点模型。通过加权融合算法，您可以将不同模型的优势结合起来，创建出风格独特、效果卓越的定制模型。

### 为什么选择这个工具？

- 🎯 **灵活的权重控制**：为每个模型分配独立权重，精确控制融合比例
- 🚀 **支持多模型融合**：不限于两个模型，可同时融合任意数量的模型
- 💾 **内存优化**：智能内存管理，支持大型 SDXL 模型融合
- ⚡ **快速验证**：内置推理功能，融合后立即生成测试图片
- 🛠️ **多种融合算法**：支持加权求和、差分添加等多种融合策略
- 📦 **SafeTensors 支持**：完整支持现代 SafeTensors 格式

## ✨ 功能特性

### 核心功能

- ✅ **多模型加权融合**：支持 2-N 个模型的智能融合
- ✅ **自动权重归一化**：自动处理权重比例，确保模型稳定性
- ✅ **VAE 兼容处理**：智能跳过 VAE 层，避免融合冲突
- ✅ **混合精度支持**：可选 FP16 半精度，节省显存和存储空间
- ✅ **跨架构融合**：支持不同通道数模型的融合（SD1.5 ↔ Inpainting）
- ✅ **即时推理验证**：融合完成后立即生成测试图片验证效果

### 支持的模型类型

- Stable Diffusion 1.5
- Stable Diffusion XL (SDXL)
- Inpainting 模型（8/9 通道）
- 自定义微调模型

### 融合算法

| 算法 | 描述 | 适用场景 |
|------|------|---------|
| `weighted-sum` | 加权求和 | 融合多个风格相近的模型 |
| `add-difference` | 差分添加 | 在基础模型上叠加特定风格 |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
CUDA >= 11.0 (推荐)
显存 >= 8GB (SDXL 建议 12GB+)
```

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/SDXLMergeModel.git
cd SDXLMergeModel

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers safetensors transformers accelerate pyyaml pillow
```

### 5 分钟快速体验

1️⃣ **准备配置文件**

创建 `my_config.yaml`：

```yaml
config:
  models:
    - path: "/path/to/model1.safetensors"
      weight: 1.0
    - path: "/path/to/model2.safetensors"
      weight: 1.0
  param:
    interpolation: "weighted-sum"
    output_dir: "/path/to/output/merged_model.safetensors"
```

2️⃣ **运行融合**

```bash
python checkpoint_merge.py my_config.yaml --half
```

3️⃣ **等待完成**

融合过程会显示详细进度，完成后自动保存模型。

## 📖 使用指南

### 基础用法

#### 方式一：使用配置文件（推荐）

```bash
python checkpoint_merge.py config.yaml --half
```

#### 方式二：融合后立即测试

```bash
python checkpoint_merge.py config.yaml \
  --half \
  --prompt "a beautiful landscape, masterpiece, best quality" \
  --image_output "./test.png" \
  --sdxl
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `yaml_path` | 配置文件路径 | 必需 |
| `--half` | 使用 FP16 半精度 | False |
| `--interpolation` | 融合算法 | weighted-sum |
| `--multiplier` | 全局倍数 | 1.0 |
| `--sdxl` | 使用 SDXL pipeline | False |
| `--prompt` | 测试提示词 | None |
| `--image_output` | 输出图片路径 | None |
| `--sampler` | 采样器名称 | Euler a |
| `--disable_torch_compile` | 禁用 Torch 编译 | False |

### 推理测试

使用 [test_promts.py](test_promts.py) 批量测试融合后的模型：

```bash
python test_promts.py
```

脚本会使用 13 个预设提示词生成测试图片，涵盖：
- 人物肖像（真实系/卡通/动漫）
- 物体渲染（汽车/眼镜/宝石）
- 场景绘制（风景/武侠/赛博朋克）

## ⚙️ 配置说明

### 配置文件结构

```yaml
config:
  models:
    # 模型列表，按顺序融合
    - path: "/path/to/model1.safetensors"
      weight: 3.0           # 权重值（会自动归一化）

    - path: "/path/to/model2.safetensors"
      weight: 4.0

    - path: "/path/to/model3.safetensors"
      weight: 1.5

  param:
    interpolation: "weighted-sum"  # 融合算法
    output_dir: "/path/to/output.safetensors"  # 输出路径
```

### 权重设置技巧

权重会自动归一化，因此您可以使用任意数值比例：

```yaml
# 这三种写法效果相同
weight: 1, 1, 1          # 比例 1:1:1
weight: 2, 2, 2          # 比例 1:1:1
weight: 0.33, 0.33, 0.34 # 比例 1:1:1

# 实际应用示例
weight: 3, 2, 1  # 模型1占50%，模型2占33%，模型3占17%
```

### 示例配置文件

项目提供了多个预设配置：

- [configXL.yaml](configXL.yaml) - 标准 SDXL 融合配置
- [configXL_Pony.yaml](configXL_Pony.yaml) - Pony 风格配置
- [configXL_log.yaml](configXL_log.yaml) - 日志记录配置
- [config_ani_log.yaml](config_ani_log.yaml) - 动漫风格配置

## 💡 示例

### 示例 1：融合两个写实风格模型

```yaml
config:
  models:
    - path: "realistic_v6.safetensors"
      weight: 1
    - path: "photorealism_v3.safetensors"
      weight: 1
  param:
    output_dir: "merged_realistic.safetensors"
```

```bash
python checkpoint_merge.py config_realistic.yaml --half
```

### 示例 2：创建多风格混合模型

```yaml
config:
  models:
    - path: "realistic_base.safetensors"
      weight: 5      # 主体：写实风格 50%
    - path: "anime_style.safetensors"
      weight: 3      # 辅助：动漫风格 30%
    - path: "artistic_filter.safetensors"
      weight: 2      # 点缀：艺术滤镜 20%
  param:
    output_dir: "hybrid_style.safetensors"
```

### 示例 3：融合后立即验证

```bash
python checkpoint_merge.py configXL.yaml \
  --half \
  --sdxl \
  --prompt "a beautiful woman, photorealistic, 8k, best quality --w 1024 --h 1024 --s 30 --l 7.5" \
  --image_output "./validation.png"
```

### 提示词参数说明

推理时支持在提示词中使用参数：

```
"your prompt here --w 1024 --h 1024 --s 50 --l 7.5 --n bad quality --d 42"
```

| 参数 | 含义 | 示例 |
|------|------|------|
| `--w` | 宽度 | `--w 1024` |
| `--h` | 高度 | `--h 768` |
| `--s` | 步数 | `--s 30` |
| `--l` | CFG Scale | `--l 7.5` |
| `--n` | 负面提示词 | `--n "blurry, bad quality"` |
| `--d` | 随机种子 | `--d 42` |
| `--t` | 每提示词图片数 | `--t 4` |

## 🔧 高级功能

### 跳过特定层

通过 `discard_weights` 参数排除特定权重层：

```bash
python checkpoint_merge.py config.yaml \
  --discard_weights "embeddings.*"
```

### 自定义融合逻辑

查看 [util/tensor.py](util/tensor.py) 了解底层张量操作，支持：
- TIES 融合策略
- DARE 掩码融合
- 自定义合并算法

## 📊 性能优化

### 减少显存占用

```bash
# 启用半精度
python checkpoint_merge.py config.yaml --half

# 禁用 Torch 编译（如遇兼容性问题）
python checkpoint_merge.py config.yaml --disable_torch_compile
```

### 加速推理

脚本默认使用 `torch.compile` 优化 UNet，首次运行会有编译时间，后续推理速度提升 20-30%。

## 🎯 最佳实践

1. **选择兼容模型**：确保所有模型基于相同基础架构（都是 SD1.5 或都是 SDXL）
2. **合理分配权重**：主要风格模型使用较大权重（如 3-5），辅助模型使用较小权重（如 0.5-2）
3. **使用半精度**：对于 SDXL 模型，建议使用 `--half` 节省 50% 存储空间
4. **跳过 VAE**：代码已自动跳过 VAE 层融合，避免图像质量下降
5. **融合后测试**：使用多样化提示词测试融合效果，确保模型泛化能力

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发计划

- [ ] 支持 LoRA 融合
- [ ] GUI 界面
- [ ] 批量融合脚本
- [ ] 融合效果对比工具
- [ ] Docker 容器化部署

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目受到以下项目启发和参考：

- [Enfugue AI](https://github.com/painebenjamin/app.enfugue.ai) - 模型合并逻辑
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - 融合算法
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 推理 Pipeline

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/SDXLMergeModel/issues)
- 发送邮件至：your.email@example.com

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个星标支持！**

Made with ❤️ for AI Artists

</div>
