"""Test script to reproduce Bug 2: CUDA graph batch size mismatch.

Run eval_model.py with concurrency >= 2 to trigger this bug.
Or send multiple concurrent requests to the server.
"""

import sglang as sgl
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import json


def main():
    os.environ['SGLANG_WATCHDOG_TIMEOUT'] = '300'

    model_path = '/share-evpfs/tj/models/MiniCPM-SALA'

    print("Testing: CUDA graph batch size mismatch (concurrent requests)")

    engine = sgl.Engine(
        model_path=model_path,
        trust_remote_code=True,
        attention_backend='minicpm_flashattn',
    )

    # Send multiple requests with different lengths to trigger batch size changes
    prompts = [
        '你好，介绍一下你自己',  # short
        '请详细解释什么是深度学习，包括其历史、原理和应用。' * 10,  # medium
        '人工智能正在改变世界。' * 5,  # short-medium
        '计算 123 * 456 = ?',  # short
    ]

    def generate(prompt):
        return engine.generate(
            prompt=prompt,
            sampling_params={'max_new_tokens': 100, 'temperature': 0.0},
        )

    print(f"Sending {len(prompts)} concurrent requests...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(generate, p) for p in prompts]
        for i, f in enumerate(futures):
            try:
                result = f.result(timeout=120)
                print(f"Request {i}: OK - {result['text'][:50]}...")
            except Exception as e:
                print(f"Request {i}: FAILED - {e}")

    engine.shutdown()
    print("Done.")


if __name__ == '__main__':
    main()
