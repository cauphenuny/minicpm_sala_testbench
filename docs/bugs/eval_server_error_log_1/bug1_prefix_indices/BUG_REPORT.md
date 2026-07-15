# Bug 1: Missing MiniCPM sparse_k1 prefix indices

## 错误信息

```
AssertionError: Missing MiniCPM sparse_k1 prefix indices for request <rid>
```

## 触发条件

- **Server 模式**下（`sglang.launch_server`），Engine 模式（`sgl.Engine`）不触发
- 序列长度 > chunked_prefill_size (8192 tokens)
- concurrency=1 也出现
- CUDA graph ON/OFF 都触发

## 复现方式

```bash
bash run.sh
```

## 完整 traceback

见 server.log
