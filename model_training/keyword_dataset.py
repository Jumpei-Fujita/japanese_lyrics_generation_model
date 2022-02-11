from janome.tokenizer import Tokenizer
import torch.nn as nn
from tqdm import tqdm
import pickle
import transformers
from transformers import T5Tokenizer
import numpy as np
import torch

class Extract_Keyword():
    def __init__(self, utanet_dataset, top_n, hinshi_list, gpu):
        self.utanet_dataset = utanet_dataset
        self.top_n = top_n
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
        self.bert = transformers.AutoModel.from_pretrained("rinna/japanese-roberta-base")
        if gpu:
            self.bert = self.bert.cuda()
        self.janome_tokenizer = Tokenizer()
        self.gpu = gpu
        self.hinshi_list = hinshi_list


    def extract_from_bert(self, lyrics, candidate):
        #bertへ入力テンソル
        lyrics_input = self.tokenizer([lyrics], truncation=True, padding=True, max_length=510, return_tensors='pt')
        candidate_input = self.tokenizer(candidate, truncation=True, padding=True, max_length=510, return_tensors='pt')

        #bertへの入力
        if self.gpu:
            candidate_tensor = self.bert.forward(input_ids=candidate_input['input_ids'].cuda(), attention_mask=candidate_input['attention_mask'].cuda())[1]
            lyrics_tensor = self.bert.forward(input_ids=lyrics_input['input_ids'].cuda(), attention_mask=lyrics_input['attention_mask'].cuda())[1]
        else:
            candidate_tensor = self.bert.forward(input_ids=candidate_input['input_ids'], attention_mask=candidate_input['attention_mask'])[1]
            lyrics_tensor = self.bert.forward(input_ids=lyrics_input['input_ids'], attention_mask=lyrics_input['attention_mask'])[1]

        #cos類似度
        with torch.no_grad():
            arglist = np.argsort(torch.cosine_similarity(candidate_tensor, lyrics_tensor).cpu().numpy())[::-1]

        return [candidate[i] for i in arglist[:self.top_n]]

    def extract_hinshi(self, text, hinshi_list):
        #hinshi_listにある品詞のみの単語を抽出（原型）
        ps = [i.part_of_speech.split(',')[0] for i in self.janome_tokenizer.tokenize(text)]
        nm = [i.base_form for i in self.janome_tokenizer.tokenize(text)]
        return [nm[i] for i in range(len(nm)) if ps[i] in hinshi_list]
    
    def extract_keyword(self):
        for i in tqdm(range(len(self.utanet_dataset))):
            artist = self.utanet_dataset[i]
            for j in range(len(artist['music_list'])):
                lyrics = ' '.join(artist['music_list'][j]['lyrics'])
                keyword_candidate = list(set(self.extract_hinshi(lyrics, self.hinshi_list)))
                self.utanet_dataset[i]['music_list'][j]['keyword'] = self.extract_from_bert(lyrics, keyword_candidate)
        
        return self.utanet_dataset

if __name__ == "__main__":


    with open("./scraper/datasets/utanet_dataset.pkl", 'rb') as f:
        utanet_dataset = pickle.load(f)
    extractor = Extract_Keyword(utanet_dataset, 6, ['名詞', '動詞', '形容詞', '形容動詞'], False)
    new_ds = extractor.extract_keyword()
    with open("./scraper/datasets/utanet_dataset_keyword.pkl","wb") as f:
        pickle.dump(new_ds, f)
    with open("./scraper/datasets/utanet_dataset_keyword.pkl","rb") as f:
        ds = pickle.load(f)
    print(ds[0]['music_list'])
                
