# trainer.py
import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        C.device = 'auto'
        C.num_workers = 4
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, valid_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.callbacks = defaultdict(list)

        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config)

        def collate_fn(batch):
            """
            batch = [(input_ids, label), (input_ids, label), ...]
            Precisamos empilhar e criar attention_mask para padding.
            """
            input_ids_list = []
            labels_list = []
            max_len = 0

            # Acha o maior comprimento real no batch
            for (inp, lbl) in batch:
                max_len = max(max_len, inp.size(0))
            max_len = min(max_len, model.block_size)

            for (inp, lbl) in batch:
                # Trunca se passar
                if inp.size(0) > max_len:
                    inp = inp[:max_len]

                # Padding se menor
                pad_size = max_len - inp.size(0)
                if pad_size > 0:
                    inp = torch.cat([inp, torch.zeros(pad_size, dtype=torch.long)], dim=0)

                input_ids_list.append(inp.unsqueeze(0))
                labels_list.append(lbl)

            input_ids_tensor = torch.cat(input_ids_list, dim=0)  # (B, T)
            labels_tensor = torch.stack(labels_list, dim=0)      # (B,)

            return input_ids_tensor, labels_tensor

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        while True:
            for batch in train_loader:
                x, labels = [t.to(self.device) for t in batch]

                # Se quiser usar attention_mask p/ ignorar PAD:
                # attention_mask = (x != 0).long()
                # logits, loss = model(x, labels=labels, attention_mask=attention_mask)
                # Caso não vá usar nada, simplesmente:
                logits, loss = model(x, labels=labels)

                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                self.loss = loss.item()
                self.trigger_callbacks('on_batch_end')

                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    break

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        print("Treinamento concluído!")
