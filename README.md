# LLM From Scratch

一个用 `PyTorch` 从零手搓的字符级 Transformer / MiniGPT 项目，用来理解大模型的核心原理。

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-Character--Level%20Transformer-0F172A)
![Built](https://img.shields.io/badge/Built-From%20Scratch-16A34A)

## 项目简介

这个项目不依赖现成的大模型封装，而是直接从底层把一个最小可运行的语言模型流程搭出来，包括：

- 从 `input.txt` 读取训练语料
- 构建字符级词表
- 手写 `encode / decode`
- 实现 `Self-Attention`、多头注意力、前馈网络和 Transformer Block
- 训练 `MiniGPTLanguageModel`
- 在命令行里进行文本生成

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 框架 | PyTorch |
| 词表方式 | 字符级 |
| `batch_size` | `16` |
| `block_size` | `32` |
| `n_embd` | `128` |
| `n_head` | `8` |
| `n_layer` | `6` |
| 优化器 | `AdamW` |
| 训练轮次 | `5000` |

## 项目结构

```text
llm/
├── haiy.py
├── input.txt
├── README.md
└── .gitignore
```

## 运行方式

安装依赖：

```bash
pip install torch
```

运行项目：

```bash
python haiy.py
```

输入 `quit` 可以退出交互。

## 仓库描述

`A hand-built character-level Transformer / MiniGPT project in PyTorch, created to understand large language models from scratch.`

## Tags

`llm`, `transformer`, `pytorch`, `gpt`, `language-model`, `nlp`, `from-scratch`, `educational`
