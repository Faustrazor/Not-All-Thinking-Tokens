# lighteval-llm

使用 LightEval 对 LLM 进行评估。

目前已囊括的数据集有：

- MATH 500
- GPQA
- AIME 24
- AIME 25

## 🛠️ 安装

项目使用 uv 作为项目管理工具，请确保安装了 uv。

```bash
cd src

uv sync
```

由于评估过程需要在 HuggingFace 平台中下载 datasets，为了防止网络问题，可以在 `~/.bashrc` 中添加如下环境变量来设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## ⚡ 运行评估脚本

使用 LightEval 编写的评估代码放在 `src/eval` 目录下，需要使用 lighteval 命令来启动。相关启动命令已编写成 shell 脚本，放在 `eval-examples` 目录下。

以运行 MATH 500 为例：

1. 修改 `eval-examples/eval-math500.sh` 脚本中的模型位置及结果输出目录。
2. 运行 shell 脚本：

> 实际上，这里建议 copy 一份 `eval-examples` 目录下的 shell 脚本，并粘贴到 scripts 目录下并进行修改，然后运行修改后放在 scripts 目录下的脚本。

```bash
# 在项目根目录下执行
source src/venv/bin/activate

bash eval-examples/eval-math500.sh
```

## 🎀 LightEval 一些用法

查看 help 文档：

```bash
lighteval --help

lighteval vllm --help
```
