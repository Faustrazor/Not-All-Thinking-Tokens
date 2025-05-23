# Not All Tokens Are What You Need in Thinking

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

Official implementation of the paper "Not All Tokens Are What You Need in Thinking"

![image](process.png)

## 📋 Overview

This repository contains the official code implementation for the paper "Not All Tokens Are What You Need in Thinking". Our work explores conditional token compression in large language models' thinking processes, demonstrating that not all tokens contribute equally to reasoning capabilities.

## 🚀 Features

- **Conditional Token Compression**: Implement reference model-based token compression strategies
- **Training Framework**: Built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning
- **Evaluation Framework**: Integrated with [LightEval](https://github.com/huggingface/lighteval) for comprehensive model evaluation
- **Ablation Studies**: Complete ablation experiment implementations

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA (for GPU training)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/Not-All-Thinking-Tokens.git
cd Not-All-Thinking-Tokens
```

2. Install basic dependencies:
```bash
pip install -r requirements.txt
```

3. Install LLaMA-Factory for training:
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

4. Install LightEval for evaluation:
```bash
pip install lighteval
```

5. Install additional dependencies for conditional compression:
```bash
pip install llmlingua
```

## 📁 Project Structure

```
Not-All-Thinking-Tokens/
├── CTS/
│   ├── train_src/          # Training scripts
│   └── test_src/           # Testing scripts
├── ablation_data_get/      # Ablation experiment code
├── CTS_data_get.py        # Conditional compression using Reference Model
├── process.png            # Process illustration
├── README.md
└── requirements.txt
```

## 🔧 Usage

### Data Preparation

#### Conditional Token Compression

Use the reference model to perform conditional compression on your dataset:

```bash
python CTS_data_get.py \
    --input_data path/to/your/dataset \
    --output_data path/to/compressed/dataset \
    --reference_model model_name \
    --compression_ratio 0.7
```

**Note**: This step requires `llmlingua` to be installed.

### Training

Navigate to the training directory and run the training scripts:

```bash
cd CTS/train_src
python train.py \
    --model_name_or_path base_model_path \
    --dataset compressed_dataset_path \
    --output_dir ./outputs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5
```

The training framework leverages [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning across various model architectures.

### Evaluation

Run the evaluation scripts using the LightEval framework:

```bash
cd CTS/test_src
python test.py \
    --model_path path/to/trained/model \
    --eval_dataset eval_dataset_name \
    --output_dir ./eval_results
```

### Ablation Studies

To reproduce the ablation experiments:

```bash
cd ablation_data_get
python ablation_experiment.py \
    --experiment_type token_selection \
    --model_path base_model_path \
    --dataset dataset_path
```

## 📊 Results

Our method demonstrates significant improvements in:
- **Efficiency**: Reduced computational overhead while maintaining performance
- **Token Selection**: Better identification of crucial thinking tokens
- **Reasoning Quality**: Enhanced reasoning capabilities with fewer tokens

## 🔬 Experiments

### Supported Models

- LLaMA series
- Qwen series  
- ChatGLM series
- Baichuan series
- And other models supported by LLaMA-Factory

### Evaluation Metrics

- Accuracy on reasoning tasks
- Token compression ratio
- Inference speed
- Memory usage

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{not_all_thinking_tokens,
  title={Not All Tokens Are What You Need in Thinking},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This work builds upon several excellent open-source projects:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - For the training framework
- [LightEval](https://github.com/huggingface/lighteval) - For the evaluation framework
- [LLMLingua](https://github.com/microsoft/LLMLingua) - For token compression techniques

## 📞 Contact

For questions or issues, please:
- Open an issue in this repository
- Contact us at [your-email@domain.com]

## 🔄 Updates

- **Latest**: Initial release with training and evaluation scripts
- **Coming Soon**: Pre-trained model weights and detailed tutorials
