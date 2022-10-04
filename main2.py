import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

text = []
for i in tqdm(range(196580, 255484+1)):
    url = 'https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&qna_seq=%d&pageIndex=4793' % (i)
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        try:
            title = soup.select_one('#content > div.boardView > div:nth-child(4) > p:nth-child(3)').get_text()
            title = re.sub('\n|\t|\r', ' ', title).strip()
            text.append(title)
        except:
            continue
    else:
        print(response.status_code)

df = pd.DataFrame({'text': text})
df.to_csv('상담사례답변모음2.csv', index=False)
print(df)
