from django.shortcuts import render, HttpResponse, redirect
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import re
import math
import time
import json
import nlpcloud
import numpy as np
from gensim import matutils 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import nlpcloud
import os
import requests

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
url = "https://api.meaningcloud.com/summarization-1.0"

def split_into_sentences(text):
    print("TEXT:",text)
    text = " " + str(text) + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
# Create your views here.
# model_name = 'google/pegasus-large'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

client = nlpcloud.Client("bart-large-cnn", "f9c60f3620a2bdba081394bd77f44445c2a019b0")

data1 = pd.read_excel(r'./data/Mili_bank_forest_final.xlsx')
summaries_data = pd.read_csv(r'./data/summaries.csv')
summaries_data['summary'].fillna('', inplace= True)
raw_data = pd.read_csv(r'./data/raw.csv',error_bad_lines=False, engine="python")

def index(request):
    return render(request, 'index.html')

def get_similar_docs(cluster,casename):
    if cluster ==  62:
        cluster = 2
    elif cluster == 6:
        cluster = 1
    elif cluster == 30:
        cluster = 0
    chosen = casename
    model = Doc2Vec.load(r"./models/d2v_old.model")
    chosen_row = data1[data1['File Name'] == casename].head(1)
    chosen_text = chosen_row['text'].values[0]
    test_data = word_tokenize(str(chosen_text).lower())
    v1 = model.infer_vector(test_data)
    
    similarities = []
    files = []
    for i in range(len(data1['text'])):
        if data1['Predicted_category'].iloc[i] == cluster and data1['File Name'].iloc[i] != casename:
            d2 = model.dv[str(i)]
            similarities.append(np.dot(matutils.unitvec(v1), matutils.unitvec(d2)))
            files.append(data1['File Name'].iloc[i])
    d2v_df = pd.DataFrame({'filename':files,'similarities':similarities})
    results = d2v_df.sort_values(by=['similarities'],ascending = False).head()
    result = results.to_dict()
    file_dict = result['filename']
    similarities_dict = result['similarities']
    file_similarities = []
    for i in file_dict:
        file_similarities.append([file_dict[i],round(similarities_dict[i],4)])
    return file_similarities

def results(request):
    print(os.getcwd())
    with open('./result_cache.json') as json_file:
        result = json.load(json_file)
    file_dict = result['filename']
    similarities_dict = result['similarities']
    file_similarities = []
    for i in file_dict:
        file_similarities.append([file_dict[i],round(similarities_dict[i],4)])
    if 'query' in request.COOKIES:
        response = render(request, 'results.html',{'file_similarities':file_similarities,'query':request.COOKIES['query'] })
        response.set_cookie(key='query', value=request.COOKIES['query'])
        return response
    return render(request, 'results.html',{'file_similarities':file_similarities})

def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        model = Doc2Vec.load(r"./models/d2v_old.model")
        test_data = word_tokenize(query.lower())
        v1 = model.infer_vector(test_data)
        similarities = []
        for i in range(len(data1['text'])):
            d2 = model.dv[str(i)]
            similarities.append( round( 100 * np.dot(matutils.unitvec(v1), matutils.unitvec(d2)),2) )
        d2v_df = pd.DataFrame({'filename':list(data1['File Name']),'similarities':similarities})
        results = d2v_df.sort_values(by=['similarities'],ascending = False).head()
        with open("./result_cache.json", "w") as fp:   #Pickling
            json.dump(results.to_dict(), fp)
        response = redirect('results')
        response.set_cookie(key='query', value=query)
        return response
    return render(request, 'search.html')

def abstract_summary(casename):
    texts = []
    if casename in set(summaries_data['File Name']):
        case_row = summaries_data[summaries_data['File Name'] == casename].head(1)
        if not case_row['summary'].values[0] == '':
            abstract = case_row['summary'].values[0]
            return abstract
    chosen_row = raw_data[raw_data['File Name'] == casename].head(1)
    if chosen_row['text'].values:
        article_text = chosen_row['text'].values[0]
        texts = sent_tokenize(article_text)
    else:
        chosen_row = data1[data1['File Name'] == casename].head(1)
        article_text = chosen_row['text'].values[0]
        texts = split_into_sentences(article_text)    
    i = 0
    steps = 40
    summary_text = ''
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
        'sentences': (20 * len(input_txt))//100
        }
        response = requests.post(url, data=payload).json()
        summary_text += response['summary'] + ' '
        i += steps
    summary_row = summaries_data[summaries_data['File Name'] == casename].head(1).index
    if summary_text:
        print(summary_row)
        summaries_data.at[summary_row,'summary'] = summary_text
        summaries_data.to_csv(r'./data/summaries.csv')
    return summary_text

def summary(request):
    if 'query' in request.COOKIES:
        query = request.COOKIES['query']
    casename = request.GET.get('filename')
    case_details = pd.read_csv(r'./data/case_details.csv')
    # summary_file = pd.read_csv(r'./data/summaries.csv')
    # summary_row = summary_file[summary_file['File Name'] == casename].head(1)
    # summary = summary_row['summary'].values[0]
    case_row = case_details[case_details['File Name'] == casename].head(1)
    if case_row['Case Name'].values:
        data_case_name =  case_row['Case Name'].values[0]
    if case_row['Involved Personell'].values:
        involved = case_row['Involved Personell'].values[0]
    if case_row['Date (Decided)'].values:
        date_decided = case_row['Date (Decided)'].values[0]
    if case_row['Court'].values:
        court = case_row['Court'].values[0]
    category = case_row['category'].values[0]
    similar_docs = get_similar_docs(category,casename)
    if category == 62:
        category = 'Military'
    elif category == 6:
        category = 'Banking'
    else:
        category = 'Environment'
    abstract = abstract_summary(casename)
    try:
        if math.isnan(court): court = ''
    except:
        pass
    try:
        if  math.isnan(date_decided): date_decided = ''
    except:
        pass
    try:
        if  math.isnan(involved): involved = ''
    except:
        pass
    try:
        if  math.isnan(data_case_name): data_case_name = ''
    except:
        pass
    
    return render(request, 'summary.html',{'abstract_summary':abstract,'similar_docs':similar_docs,'data_case_name':data_case_name,'involved':involved,'date_decided':date_decided,'court':court,'category':category,'query':query})
