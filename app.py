import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration

from flask import Flask,render_template,url_for
from flask import request as req
#import nltk
from newspaper import Article
import urllib.request

from bs4 import BeautifulSoup
from hashlib import new
from ntpath import join

def header(url):
    html = urllib.request.urlopen(url)
# parsing the html file
    htmlParse = BeautifulSoup(html, 'html.parser')
#for extracting heading
    target1=htmlParse.find('div', {'class':"innner-page-main-about-us-content-right-part"})
    target2=target1.find('h2')
    mainheading=target2.get_text()
    return(mainheading)

def datetime(url):
    html = urllib.request.urlopen(url)
# parsing the html file
    htmlParse = BeautifulSoup(html, 'html.parser')
    datetime1=htmlParse.find('div', {'class':"ReleaseDateSubHeaddateTime text-center pt20"})
    datetime2=datetime1.get_text()
    return(datetime2)

def summarize(url):
    html = urllib.request.urlopen(url)
    # parsing the html file
    htmlParse = BeautifulSoup(html, 'html.parser')
    target=htmlParse.find('div')
    paras=target.find_all('p')

    arr = []    

    for para in paras:
        out=para.get_text()
        arr.append(out)

    list2 = " ".join(arr)
    

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    ARTICLE_TO_SUMMARIZE = list2 

    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=200, max_length=300)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #return(summary)

    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(input_text,num_return_sequences):
        batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    # Paragraph of text
    context = summary

    # Takes the input paragraph and splits it into a list of sentences
    from sentence_splitter import SentenceSplitter, split_text_into_sentences

    splitter = SentenceSplitter(language='en')

    sentence_list = splitter.split(context)
    #sentence_list

    # Do a for loop to iterate through the list of sentences and paraphrase each sentence in the iteration
    paraphrase = []

    for i in sentence_list:
        a = get_response(i,1)
        paraphrase.append(a)

    # This is the paraphrased text
    #paraphrase

    paraphrase2 = [' '.join(x) for x in paraphrase]
    #paraphrase2

    # Combines the above list into a paragraph
    paraphrase3 = [' '.join(x for x in paraphrase2) ]
    paraphrased_text = str(paraphrase3).strip('[]').strip("'")
    #paraphrased_text

    return(paraphrased_text)




app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def Index():
    if req.method == "POST":
        url = req.form.get("url")
        date_content = datetime(url)
        url_content = summarize(url)
        header_content = header(url)
        return render_template("main1.html",value1=header_content,value3=date_content,value2=url_content)
        
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)