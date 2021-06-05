import torch
import torch.nn as nn

from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from set_seed import set_seed

set_seed()

# Create the BertClassfier class
class BertClassifier(nn.Module):

    def __init__(self, model_name, dropout, freeze_bert=False, classes=3):

        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 200, classes

        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits