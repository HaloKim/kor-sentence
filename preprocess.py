from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer-trained.json")

encoded = tokenizer.encode("배가 아파서 집에 가고 싶어요.")
print(tokenizer.decode(encoded.ids))
