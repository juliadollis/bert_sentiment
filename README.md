# BERT - CLASSIFICADOR DE SENTIMENTOS - POSITIVO E NEGATIVO

Este é um projeto de implementação de um modelo estilo BERT que utiliza o dataset IMDB para classificação de sentimentos (positivo e negativo).

## Instalação

Siga os passos abaixo para utilizar o projeto no Google Colab:

1.Abra o Google Colab e crie um Novo Projeto


2.Na barra superior selecione "Tempo de Execução" e depois "Alterar tipo de ambiente de execução" e escolha alguma GPU disponível


3. Clone o repositório:
    ```bash
    #1
    !git clone https://github.com/juliadollis/bert_sentiment.git
    %cd bert_sentiment
    ```

    
3. Instale as dependências:
    ```bash
    #2
    !pip install -r requirements.txt
    ```

    
4. Execute o projeto:
    ```bash
    
    # 3) Imports
    import torch
    from datasets import load_dataset
    from torch.utils.data import Dataset
    from sklearn.metrics import accuracy_score
    from bpe import BPETokenizer
    from model import BERT
    from trainer import Trainer
    from utils import CfgNode, set_seed
    
    # 4) Carregar IMDB via HuggingFace
    dataset = load_dataset('imdb')
    
    # 5) Criar classes de dataset
    tokenizer = BPETokenizer()
    MAX_LENGTH = 128
    
    class IMDBDataset(Dataset):
        def __init__(self, split):
            self.data = dataset[split]

        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            text = self.data[idx]['text']
            label = self.data[idx]['label']  # 0 ou 1
            input_ids = tokenizer(text, return_tensors='pt').squeeze(0)
            # (Não fazemos PAD aqui, pois o collate_fn no trainer.py cuida disso)
            # Apenas truncamos se ultrapassar block_size
            if input_ids.size(0) > MAX_LENGTH:
                input_ids = input_ids[:MAX_LENGTH]
            return input_ids, torch.tensor(label, dtype=torch.long)
    
            train_dataset = IMDBDataset('train')
            test_dataset  = IMDBDataset('test')
            
        # 6) Configurar modelo
        cfg = BERT.get_default_config()
        cfg.vocab_size = 50257
        cfg.block_size = MAX_LENGTH
        cfg.n_layer = 4
        cfg.n_head = 4
        cfg.n_embd = 256
        cfg.num_labels = 2
        # Se quiser, pode manter model_type='bert' ou deixar None
        cfg.model_type = 'bert'
        
        model = BERT(cfg)  
        
        # 7) Configurar Trainer
        trainer_config = Trainer.get_default_config()
        trainer_config.max_iters = 80000  # Ajuste conforme seu tempo/GPU
        trainer_config.batch_size = 8
        trainer_config.learning_rate = 3e-4
        trainer_config.num_workers = 2  # no Colab, tente deixar pequeno p/ evitar avisos

        trainer = Trainer(trainer_config, model, train_dataset, valid_dataset=test_dataset)

        # 8) Treinar
        trainer.run()
        
        # 9) Avaliar no test_dataset
        model.eval()
        all_preds = []
        all_labels = []
        
        # Precisamos do mesmo collate_fn do trainer
        def collate_fn(batch):
            input_ids_list = []
            labels_list = []
            max_len = 0
            for (inp, lbl) in batch:
                max_len = max(max_len, inp.size(0))
            max_len = min(max_len, model.block_size)
            for (inp, lbl) in batch:
                if inp.size(0) > max_len:
                    inp = inp[:max_len]
                else:
                    pad_size = max_len - inp.size(0)
                    if pad_size > 0:
                        inp = torch.cat([inp, torch.zeros(pad_size, dtype=torch.long)], dim=0)
                input_ids_list.append(inp.unsqueeze(0))
                labels_list.append(lbl)
            input_ids_tensor = torch.cat(input_ids_list, dim=0)
            labels_tensor = torch.stack(labels_list, dim=0)
            return input_ids_tensor, labels_tensor
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in test_loader:
                x, labels = [t.to(trainer.device) for t in batch]
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"Acurácia no teste: {acc:.4f}")
    ```

5. Realize inferências:
    ```bash
    # 10) Inferência em texto novo
    def infer_sentiment(model, tokenizer, text):
        model.eval()
        with torch.no_grad():
            x = tokenizer(text, return_tensors='pt')
            if x.size(1) > MAX_LENGTH:
                x = x[:, :MAX_LENGTH]
            # pad se quiser
            length = x.size(1)
            if length < MAX_LENGTH:
                pad_size = MAX_LENGTH - length
                x = torch.cat([x, torch.zeros((1, pad_size), dtype=torch.long)], dim=1)
            x = x.to(next(model.parameters()).device)
            logits, _ = model(x)
            pred = torch.argmax(logits, dim=1).item()
        return "POS" if pred == 1 else "NEG"
    
    ex = "ameii"
    print(ex, "->", infer_sentiment(model, tokenizer, ex))
    
    ex2 = "odiei esse filme"
    print(ex2, "->", infer_sentiment(model, tokenizer, ex2))
    ```
