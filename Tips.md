# Tips and Traces

## Tips
- 编译`sglang`时使用最新版本（3.13）的 Python 疑似会出现找不到 Rust 编译器的问题
    - Python 3.11
- H200 最低支持的 CUDA 版本为 12.4，不支持 gcc-13/g++-13，需要手动软链接为gcc-12
    - 如果使用 conda，方法为：
    - `ls /usr/bin | grep gcc`
    - `cd $(dirname $(which python))`
    - `ln -s /usr/bin/gcc-12 gcc`
    - `ln -s /usr/bin/g++-12 g++`
- 更改 pytorch 编译时需要的 CUDA 版本
    - build/CMakeCache.txt -> CMAKE_CUDA_COMPILER:STRING=/usr/local/cuda-12.9/bin/nvcc
- 自己编译`pytorch 2.8.0`后运行`sglang`，可能需要一个对应版本的`torchvision`，但是`pip`会检查`torchvision`的依赖是否存在（官方的），不存在会帮你下 pytorch
    - `pip install torchvision==0.23.0 --no-deps`
- flashinfer show-config 返回：Module registration failed: No module named 'nvidia.nvshmem'
    - ❌ Module registration failed: No module named 'nvidia.nvshmem'
    - `pip install nvidia-nvshmem-cu12`
- https://github.com/sgl-project/sglang/issues/8661

## Traces

### PyTorch 调用 Trace
```shell
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 378, in __init__
[rank0]:     self.capture()                         
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 497, in capture
[rank0]:     ) = self.capture_one_batch_size(bs, forward)
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 676, in capture_one_batch_size
[rank0]:     run_once()                             
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 663, in run_once
[rank0]:     logits_output_or_pp_proxy_tensors = forward(
[rank0]:                                         ^^^^^^^^
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/utils/_contextlib.py", line 120, in decorate_context
[rank0]:     return func(*args, **kwargs)           
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^           
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/models/llama.py", line 469, in forward
[rank0]:     hidden_states = self.model(            
[rank0]:                     ^^^^^^^^^^^            
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)   
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/models/llama.py", line 342, in forward
[rank0]:     hidden_states, residual = layer(       
[rank0]:                               ^^^^^^       
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)   
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/models/llama.py", line 266, in forward
[rank0]:     hidden_states = self.self_attn(            
[rank0]:                     ^^^^^^^^^^^^^^^            
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)    
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)       
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/models/llama.py", line 194, in forward
[rank0]:     qkv, _ = self.qkv_proj(hidden_states)      
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^      
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)    
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
[rank0]:   File "/home/yzy/rebuild-pytorch/pytorch/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)       
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/layers/linear.py", line 427, in forward
[rank0]:     output_parallel = self.quant_method.apply(self, input_, bias)
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yzy/rebuild-pytorch/sglang/python/sglang/srt/layers/quantization/unquant.py", line 131, in apply
[rank0]:     return F.linear(x, layer.weight, bias) 
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
```