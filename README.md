# FlexMPS
## General

```shell
# env
source <CONDA_INSTALL_DIR>/bin/activate
conda create -y -n <CONDA_NAME>
conda activate <CONDA_NAME>
```

```shell
# code
git clone https://github.com/yangzy723/FlexMPS.git
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

## SgLang
```shell
cd sglang-v0.5.4
pip install --upgrade pip
# 这个只是在安装 python 包，对 python 代码的修改通过以下更新生效
python -m pip install -e "python"
pip uninstall torch

# 若修改了 flashinfer 等第三方库的代码，请务必注意 sgl-kernel/ 下 CMakeLists.txt 的第三方库路径，以及是否需要重新编译 sgl-kerenl/
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
```

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

---

## Run
```shell
# 启动拦截服务端
cd server
make
./scheduler

# 启动推理客户端 benchmark
export CUDA_VISIBLE_DEVICES=0
python -m sglang.bench_one_batch --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ --batch-size 64 --input-len 512 --mem-fraction-static 0.6 --disable-cuda-graph

# nsys
nsys profile \
  --trace=cuda,nvtx,osrt \
  --python-backtrace=cuda \
  --cudabacktrace=kernel:0 \
  --python-sampling=true \
  --gpu-metrics-devices=all \
  -o sglang_report \
  --force-overwrite=true \
  python -m sglang.bench_one_batch \
  --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ \
  --batch-size 64 \
  --input-len 512 \
  --mem-fraction-static 0.6 \
  --disable-cuda-graph
```

## Prefill-Decode  Test
```shell
# 开启 MPS
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

# 启动拦截服务端
cd server
make
./scheduler

# New Terminal, Prefill Node
export UNIQUE_ID=1
python -m sglang.launch_server \
   --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ \
   --port 30001 \
   --host 0.0.0.0 \
   --mem-fraction-static 0.4

# New Terminal, Decode Node
export UNIQUE_ID=2
python -m sglang.launch_server \
   --model-path /data/datasets/models-hf/Llama-3.1-8B-Instruct/ \
   --port 30002 \
   --host 0.0.0.0 \
   --mem-fraction-static 0.4

# 启动压力测试程序
cd benchmark/test-pd
bash run.sh
```

## Versions

|模块名称       | 版本  |
|--------------|------|
|cuda          |12.9  |
|python        |3.11  |
|torch         |2.8.0 |
|sglang        |0.5.4 |
|flashinfer    |0.4.1 |
