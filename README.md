# Rebuilt-PyTorch

## General

```shell
# env
source <CONDA_INSTALL_DIR>/bin/activate
conda create -y -n <CONDA_NAME>
conda activate <CONDA_NAME>
```

```shell
# code
git clone https://github.com/yangzy723/Rebuilt-PyTorch.git
cd Rebuilt-PyTorch
git submodule sync
git submodule update --init --recursive
```

```shell
# Init
# https://github.com/pytorch/pytorch/tree/v2.8.0
# https://github.com/flashinfer-ai/flashinfer/tree/v0.4.1
# https://github.com/sgl-project/sglang/tree/v0.5.4
```
---

## PyTorch
```shell
cd pytorch-v2.8.0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

## FlashInfer
```shell
cd flashinfer-v0.4.1
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install --no-build-isolation -e . -v --no-deps

# 一些无用编译命令...
# export MAX_JOBS=64
# export FLASHINFER_CUDA_ARCH_LIST=9.0
# python -m flashinfer.aot
# python -m build --no-isolation --wheel -o ../build/
```

参考：
- https://docs.flashinfer.ai/installation.html
- https://www.cnblogs.com/iLex/p/19036981

## SgLang
```shell
cd sglang-v0.5.4
pip install --upgrade pip
# 这个只是在安装 python 包，对 python 代码的修改通过以下更新生效
python -m pip install -e "python"
pip uninstall torch
```

若修改了 flashinfer 等第三方库的代码，请务必注意 sgl-kernel/ 下 CMakeLists.txt 的第三方库路径，以及是否需要重新编译 sgl-kerenl/

```shell
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
```

## Run
```shell
export CUDA_VISIBLE_DEVICES=1
python -m sglang.bench_one_batch --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ --batch-size 64 --input-len 512 --mem-fraction-static 0.6 --disable-cuda-graph

# nsys
nsys profile --trace=cuda --sample=process-tree -   -cudabacktrace=kernel:0 -o output_report python -m sglang.bench_one_batch --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ --batch-size 64 --input-len 512 --mem-fraction-static 0.6 --disable-cuda-graph
```

## Tips
- 编译`sglang`时使用最新版本（3.13）的 Python 疑似会出现找不到 Rust 编译器的问题
    - Python 3.11
- H200 最低支持的 CUDA 版本为 12.4，不支持 gcc-13/g++-13，需要手动软链接为gcc-12
    - 如果使用 conda，方法为：
    - ls /usr/bin | grep gcc
    - cd $(dirname $(which python))
    - ln -s /usr/bin/gcc-12 gcc
    - ln -s /usr/bin/g++-12 g++
- 更改 pytorch 编译时需要的 CUDA 版本
    - build/CMakeCache.txt -> CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda-12.9/bin/nvcc
- 自己编译`pytorch 2.8.0`后运行`sglang`，可能需要一个对应版本的`torchvision`，但是`pip`会检查`torchvision`的依赖是否存在（官方的），不存在会帮你下 pytorch
    - pip install torchvision==0.23.0 --no-deps
- flashinfer show-config 返回：Module registration failed: No module named 'nvidia.nvshmem'
    - ❌ Module registration failed: No module named 'nvidia.nvshmem'
    - pip install nvidia-nvshmem-cu12
- https://github.com/sgl-project/sglang/issues/8661

## Versions

|模块名称       | 版本  |
|--------------|--------|
|cuda          |12.9  |
|python        |3.11  |
|flashinfer    |0.4.1 |
|torch         |2.8.0 |
|sglang        |0.5.4 |
