from transformers import  RobertaModel, RobertaTokenizer, BertConfig
import torch.nn as nn


MODEL_TYPE = 'microsoft/codebert-base'
TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_TYPE)

LEAENING_RATE = 1e-6
BATCH_SIZE = 256
N_EPOCHS = 100

class CodeBert(nn.Module):
    def __init__(self, model_type, tokenizer):
        super(CodeBert, self).__init__()

        bert_conf = BertConfig.from_pretrained(model_type, output_hidden_states=False, output_attentions=True)
        bert_conf.vocab_size = tokenizer.vocab_size

        self.bert = RobertaModel.from_pretrained(model_type, config=bert_conf, ignore_mismatched_sizes=True)
        self.fc = nn.Linear(bert_conf.hidden_size, out_features=1)

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        h = out['pooler_output']
        a = out['attentions']
        h = nn.ReLU()(h)
        h = self.fc(h)
        # h = h[:, 0]
        a = a[-1].sum(1)[:, 0, :]
        return h, a
