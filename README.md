# Foundation Model — Pre-trained Language Models & GPT

A hands-on notebook covering the fundamentals of large language models, from tokenisation and evaluation metrics to GPT-2 text generation with multiple decoding strategies.

---

## Table of Contents

1. [Overview](#overview)
2. [Topics Covered](#topics-covered)
   - [Byte Pair Encoding (BPE)](#1-byte-pair-encoding-bpe)
   - [Perplexity](#2-perplexity)
   - [GPT-2 Text Generation](#3-gpt-2-text-generation)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Results Summary](#results-summary)

---

## Overview

This notebook explores pre-trained language models with a focus on GPT-2. It covers tokenisation via Byte Pair Encoding, language model evaluation using perplexity, and practical text generation using the HuggingFace `transformers` library. Three decoding strategies are implemented and compared — beam search, top-k sampling, and top-p (nucleus) sampling.

---

## Topics Covered

### 1. Byte Pair Encoding (BPE)

Manual BPE walkthrough using the sentence:

```
Foundation of the foundation models
```

The exercise applies **3 merge steps** and tracks the evolving vocabulary at each step, illustrating how subword tokenisation works in practice.

**Key concept:** BPE starts with a character-level vocabulary and iteratively merges the most frequent adjacent symbol pairs, reducing out-of-vocabulary words while keeping the vocabulary size manageable.

---

### 2. Perplexity

Evaluates two language models (A and B) on the sentence:

```
The students open their books unwillingly
```

**Word probabilities:**

| Word        | Model A | Model B |
|-------------|---------|---------|
| The         | 0.25    | 0.10    |
| students    | 0.30    | 0.05    |
| open        | 0.20    | 0.10    |
| their       | 0.10    | 0.10    |
| books       | 0.10    | 0.05    |
| unwillingly | 0.05    | 0.01    |

**Formula used:**

$$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_1, \ldots, w_{i-1}) \right)$$

**Results:**

| Model | Perplexity | Verdict |
|-------|-----------|---------|
| A     | **7.15**  | ✅ Better |
| B     | 18.48     | ❌ Worse  |

> Model A performs better — lower perplexity indicates higher predictive confidence for the given sentence.

---

### 3. GPT-2 Text Generation

Uses `GPT2LMHeadModel` and `GPT2Tokenizer` from HuggingFace for interactive text generation.

#### Baseline — Beam Search

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_text(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        input_ids=inputs,
        max_length=1024,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Task — Compare Decoding Strategies

Three generation methods are implemented and compared side-by-side:

| Method | Parameters | Characteristic |
|--------|-----------|----------------|
| **Beam Search** | `num_beams=5`, `no_repeat_ngram_size=2` | Deterministic; coherent but sometimes repetitive |
| **Top-k Sampling** | `top_k=50`, `temperature=0.7` | Stochastic; more creative and varied |
| **Top-p (Nucleus) Sampling** | `top_p=0.92`, `temperature=0.7` | Stochastic; dynamically adjusts token pool |

```python
# Compare all three methods on a single prompt
prompt = input("Enter your prompt: ")

print("Beam Search:")
print(generate_text(prompt, method='beam'))

print("\nTop-k Sampling:")
print(generate_text(prompt, method='topk'))

print("\nTop-p Sampling:")
print(generate_text(prompt, method='topp'))
```

#### Bonus — Inspect Next-Token Probabilities

A utility function visualises the top-k most likely next tokens at any position in the sequence:

```python
show_top_predictions("The future of AI will", top_k=5)
```

Sample output:
```
Top 5 predictions for: 'The future of AI will'
1. ' be':   23.45%
2. ' have':  8.12%
3. ' not':   6.78%
...
```

---

## Getting Started

### Prerequisites

```bash
pip install transformers torch
```

### Clone and Run

```bash
git clone <your-repo-url>
cd <repo-folder>
jupyter notebook GPT-Model.ipynb
```

---

## Usage

1. Open `GPT-Model.ipynb` in Jupyter or JupyterLab.
2. Run cells sequentially — the BPE and perplexity sections are self-contained.
3. For the GPT-2 section, enter a text prompt when prompted and observe how each decoding strategy generates different continuations.

---

## Results Summary

| Task | Key Result |
|------|-----------|
| BPE (3 merge steps) | Vocabulary built iteratively by merging most-frequent symbol pairs |
| Perplexity | Model A (PPL = 7.15) outperforms Model B (PPL = 18.48) |
| Beam Search | Deterministic, coherent output with n-gram repetition prevention |
| Top-k Sampling | Creative, diverse output by sampling from top-50 tokens |
| Top-p Sampling | Adaptive sampling from the minimum token set covering 92% probability mass |

---

## References

- Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised). OpenAI.
- HuggingFace `transformers` documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Sennrich, R. et al. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL.
