import requests
import time
from nltk.tokenize import sent_tokenize
import pandas as pd

url = "https://api.meaningcloud.com/summarization-1.0"
data1 = pd.read_csv(r'../data/raw.csv',error_bad_lines=False, engine="python")
summaries_data = pd.read_csv(r'../data/summaries.csv')

for j in range(194,len(data1)):
    text = data1['text'].iloc[j]
    casename = data1['File Name'].iloc[j]
    if len(summaries_data[summaries_data['File Name'] == casename].head(1).index) != 0:
        summary_row = summaries_data[summaries_data['File Name'] == casename].head(1).index[0]
    if not text:
        continue
    texts = sent_tokenize(text)
    i = 0
    steps = 50
    summary_text = ''
    print(j)
    print(len(texts))
    print(casename)
    while i < len(texts):
        print(f"{round(100 * (i / len(texts)),2)}%")
        if i+steps < len(texts):
            input_txt = ''.join(texts[i:i + steps])
            # 
        else:
            input_txt = ''.join(texts[i:])
        payload={
        'key': '1a7b0a96fb81d6c881ddc7be188098ab',
        'txt': input_txt,
        'sentences': (40 * len(input_txt))//100
        }
        response = requests.post(url, data=payload).json()
        summary_text += response['summary'] + ' '
        print("Sleep time")
        time.sleep(1)
        i += steps
    if len(summaries_data[summaries_data['File Name'] == casename].head(1).index) != 0:
        summary_row = summaries_data[summaries_data['File Name'] == casename].head(1).index[0]
        summaries_data.at[summary_row,'summary'] = summary_text
    else:
        df2 = {'File Name': casename, 'summary': summary_text}
        summaries_data = summaries_data.append(df2, ignore_index = True)
    summaries_data.to_csv(r'../data/summaries.csv', index=False)