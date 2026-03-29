from pathlib import Path

from tokenizers import Encoding, Tokenizer

tokenizer_path = Path(__file__).with_name("Qwen3Tokenizer.json")

tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))

text = "你好，世界！"
encoding: Encoding = tokenizer.encode(text)
decoded_text = tokenizer.decode(encoding.ids)

tokens_as_str = [tokenizer.decode([token_id]) for token_id in encoding.ids]

print(f"{text=}")
print(f"{encoding.ids=}")
print(f"{decoded_text=}")
print(f"{tokens_as_str=}")


# text='你好，世界！'
# encoding.ids=[108386, 3837, 99489, 6313]
# decoded_text='你好，世界！'
# tokens_as_str=['你好', '，', '世界', '！']
