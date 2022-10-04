from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
print(tokenizer.encode("한진 정보 통신 AILab 김광륜 사원 입니다.").ids)
print(tokenizer.decode(tokenizer.encode("한진 정보 통신 AILab 김광륜 사원 입니다.").ids))

print(tokenizer.encode("'미래의어느때에가서는'을뜻하는부사'언젠가'를강조하기위하여,'강조'의뜻을나타내는보조사'는'을붙여'언젠가는'과같이쓸수있습니다.").tokens)
print("'미래의 어느 때에 가서는'을 뜻하는 부사 '언젠가'를 강조하기 위하여, '강조'의 뜻을 나타내는 보조사 '는'을 붙여 '언젠가는'과 같이 쓸 수 있습니다.".split(' '))
