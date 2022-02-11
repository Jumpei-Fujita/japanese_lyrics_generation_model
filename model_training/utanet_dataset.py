import numpy as np
from torch.utils.data import Dataset


class UtanetDataset(Dataset):
    def __init__(self, utanet_dataset, tokenizer, max_keyword_num, max_length):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(len(utanet_dataset)):
            for m in utanet_dataset[i]['music_list']:
                for l in m['lyrics']:
                    self.data.append({'keyword':m['keyword'], 'lyric':l})
        self.max_keyword_num = max_keyword_num
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def select_keyword(self, keywords):
        size = np.random.randint(self.max_keyword_num)
        if size != 0:
            return ','.join(np.random.choice(keywords, size=size, replace=False))
        else:
            return ''
    
    def __getitem__(self, idx):
        music = self.data[idx]
        keyword = self.select_keyword(music['keyword'])

        input_str = keyword + '/' + music['lyric']
        inputs = tokenizer(input_str, return_tensors='pt', max_length=self.max_length, padding="max_length", truncation=True)
        return inputs['input_ids'][0][:-1], inputs['attention_mask'][0][:-1], inputs['input_ids'][0][1:]
        
