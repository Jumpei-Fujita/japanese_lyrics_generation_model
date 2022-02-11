import argparse
from utanet_dataset import UtanetDataset
from model import GPT2_Lyrics
import pickle
import transformers
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mk', '--max_keyword_num', help='キーワードを考慮する最大数', type=int, required=True)
    parser.add_argument('-ml', '--max_length', help='学習する系列の最大長', type=int, required=True)
    parser.add_argument('-tr', '--train_rate', help='訓練データの割合(0~1)', type=float, required=True)
    parser.add_argument('-bs', '--batch_size', type=int, required=True)
    parser.add_argument('-mn', '--model_name', type=str, required=True)
    args = parser.parse_args()

    with open("./scraper/datasets/utanet_dataset_keyword.pkl","rb") as f:
        utanet_dataset = pickle.load(f)        
    tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
    
    dataset = UtanetDataset(utanet_dataset, tokenizer, args.max_keyword_num, args.max_length)

    n_train = int(args.train_rate * len(dataset))
    n_val = len(dataset) - n_train
    if n_val <= 0:
        print('n_trainは0~1の値で設定してください')
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    model = GPT2_Lyrics(tokenizer.pad_token_id, train_dataset, validation_dataset, args.batch_size)
    torch.manual_seed(0)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss")], max_epochs=200)
    trainer.fit(model)
    model_cpu = model.cpu()
    model_path = './model_training/saved_model/' + args.model_name + '.pt'
    torch.save(model_cpu.state_dict(), model_path)




