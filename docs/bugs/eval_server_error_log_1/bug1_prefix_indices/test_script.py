"""Test script to reproduce Bug 1: Missing sparse_k1 prefix indices.

Set NUMBER >= 250 to trigger chunked prefill on long sequences.
"""

import sglang as sgl
import os


def main():
    os.environ['SGLANG_WATCHDOG_TIMEOUT'] = '300'

    model_path = '/share-evpfs/tj/models/MiniCPM-SALA'

    print("Testing: Missing sparse_k1 prefix indices (long sequence chunked prefill)")

    engine = sgl.Engine(
        model_path=model_path,
        trust_remote_code=True,
        attention_backend='minicpm_flashattn',
        port=30000,
    )

    # NUMBER=300 -> ~15000 tokens, triggers chunked prefill (> 8192)
    NUMBER = 300
    text = '人工智能正在改变世界。深度学习使机器能够从数据中学习。自然语言处理让计算机理解人类语言。计算机视觉让机器看懂图像。' * NUMBER
    prompt = text + '请总结上文主题：'
    print(f'Prompt chars: {len(prompt)} (~{len(prompt)//1.5:.0f} tokens estimated)')

    print("Generating...")
    output = engine.generate(
        prompt=prompt,
        sampling_params={'max_new_tokens': 30, 'temperature': 0.0},
    )
    print(f'Output: {output["text"][:200]}')

    engine.shutdown()
    print("Done.")


if __name__ == '__main__':
    main()
