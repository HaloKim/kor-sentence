import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm import tqdm

text = []
for i in tqdm(range(5552, 9301+1)):
    url = 'https://www.korean.go.kr/front/mcfaq/mcfaqView.do?mn_id=217&mcfaq_seq=%d&pageIndex=1' % (i)
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        try:
            title = soup.select_one('#content > div.boardView > div.body > div').get_text()
            title = re.sub('[\[답변\]]', ' ', title)
            title = re.sub('\n|\t|\r', ' ', title).strip()
            print(title)
            text.append(title)
        except:
            continue
    else:
        print(response.status_code)

df = pd.DataFrame({'row' : range(1, len(text)+1),
                   'text' : text})
df.to_csv('상담사례답변모음.csv', index=False)
print(df)
