import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers

class GPT2_Lyrics(pl.LightningModule):
    def __init__(self, pad_token_id, train_dataset, validation_dataset, batch_size=40):
        super(GPT2_Lyrics, self).__init__()
        self.pad_token_id = pad_token_id
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained("colorfulscoop/gpt2-small-ja")
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
    def forward(self, input_ids, attention_mask):
        output = self.gpt2.forward(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output

    def reshape_tensors(self, output, label):
        final_output = output.contiguous().view(output.size(0)*output.size(1), -1)
        final_label = label.contiguous().view(label.size(0)*label.size(1))
        return final_output, final_label
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch

        output = self.forward(input_ids, attention_mask)
        final_output, final_label = self.reshape_tensors(output, label)

        loss = F.cross_entropy(final_output, final_label, ignore_index=tokenizer.pad_token_id)
        results = {'loss':loss}
        return loss
    
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_dataset, self.batch_size)
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self.forward(input_ids, attention_mask)
        final_output, final_label = self.reshape_tensors(output, label)
        loss = F.cross_entropy(final_output, final_label, ignore_index=tokenizer.pad_token_id)
        results = {'val_loss':loss}
        return loss