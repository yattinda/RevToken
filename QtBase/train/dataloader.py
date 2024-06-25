import torch
import numpy as np

class InlineChangeDataset():
    def __init__(self, toks, targets):
        self.toks = toks
        self.targets = targets

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, item):
        tok = self.toks[item]
        target = self.targets[item]

        input_ids = torch.tensor(tok["input_ids"])
        attention_mask = torch.tensor(tok["attention_mask"])
        # token_type_ids = torch.tensor(tok["token_type_ids"])
        target = torch.tensor(target).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "target": target,
        }

class InlineChangeDataset_for_token():
    def __init__(self, toks, shaped_code, targets):
        self.toks = toks
        self.targets = targets
        self.shaped_code = shaped_code

    def __len__(self):
        return len(self.toks)

    def __getitem__(self, item):
        tok = self.toks[item]
        target = self.targets[item]
        shaped_code = self.shaped_code[item]

        input_ids = torch.tensor(tok["input_ids"])
        attention_mask = torch.tensor(tok["attention_mask"])
        # token_type_ids = torch.tensor(tok["token_type_ids"])
        target = torch.tensor(target).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "target": target,
            'shaped_code':shaped_code
        }

def train_loop(train_dataloader, model, optimizer, device, tqdm, loss_fn):
    losses = []
    model.train()
    optimizer.zero_grad()
    for n_iter, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        # token_type_ids = d["token_type_ids"].to(device)
        target = d["target"].to(device)
        target = target.unsqueeze(1)

        output, _ = model(input_ids, attention_mask,)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    return losses

def test_loop(test_dataloader, model, device, tqdm, loss_fn):
    losses, predicts = [], []
    model.eval()
    for n_iter, d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        # token_type_ids = d["token_type_ids"].to(device)
        target = d["target"].to(device)
        target = target.unsqueeze(1)

        with torch.no_grad():
            output, _ = model(input_ids, attention_mask,)

        loss = loss_fn(output, target)
        losses.append(loss.item())
        predicts += output.sigmoid().cpu().tolist()

    return predicts, np.array(losses).mean()