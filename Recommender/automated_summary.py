import pandas as pd
import nlpcloud
from nltk.tokenize import sent_tokenize
import time

data1 = pd.read_csv(r'../data/raw.csv',error_bad_lines=False, engine="python")
summaries_data = pd.read_csv(r'../data/summaries.csv')
client = nlpcloud.Client("bart-large-cnn", "0353d3846980d048ff8a4958e71fe558bb0d6ab6")

for j in range(95,len(data1)):
    text = data1['text'].iloc[j]
    casename = data1['File Name'].iloc[j]
    texts = sent_tokenize(text)
    i = 0
    steps = 40
    summary_text = ''
    print(j)
    print(len(texts))
    print(casename)
    while i < len(texts):
        print(f"{round(100 * (i / len(texts)),2)}%")
        if i+steps < len(texts):
            res = client.summarization(''.join(texts[i:i + steps]))
            summary_text += res['summary_text'] + ' '
        else:
            res = client.summarization(''.join(texts[i:]))
            summary_text += res['summary_text'] + ' '
        print("Sleep time")
        time.sleep(20)
        i += steps
    summary_row = summaries_data[summaries_data['File Name'] == casename].head(1).index
    summaries_data.at[summary_row,'summary'] = summary_text
    summaries_data.to_csv(r'../data/summaries.csv')