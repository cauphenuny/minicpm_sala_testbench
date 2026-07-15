# Bug 2: CUDA graph batch size mismatch

## 错误信息

```
RuntimeError: The size of tensor a (N) must match the size of tensor b (M) at non-singleton dimension 0
```

发生在 `backend.py:_replay_cuda_graph_metadata` 的 `history_compress_token_nums[:real_bs].copy_()` 调用。

## 触发条件

- CUDA graph ON (默认)
- concurrency >= 2
- 一个请求完成导致 batch size 变化时触发

## 复现方式

```bash
bash run.sh
```

## 完整 traceback

见 server.log
