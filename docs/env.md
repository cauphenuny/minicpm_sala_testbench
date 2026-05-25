install

```bash
uv add --editable "sglang[all] @ sglang/python"
uv add "infllm_v2 @ infllmv2_cuda_impl"
uv add "sparse_kernel_extension @ sparse_kernel"
```

run sglang server

```bash
bash launch_server.sh
```