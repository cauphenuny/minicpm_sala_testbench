rm -r sglang/3rdparty/sparse_kernel/*.so sglang/3rdparty/sparse_kernel/*.egg-info
rm -r sglang/3rdparty/infllmv2_cuda_impl/infllm_v2/*.so sglang/3rdparty/infllmv2_cuda_impl/*.egg-info
rm -r sglang/python/*.egg-info
uv sync \
    --reinstall-package infllm-v2 \
    --reinstall-package sparse-kernel-extension \
    --reinstall-package sgl-kernel \
    --reinstall-package sglang \
    --no-build-isolation