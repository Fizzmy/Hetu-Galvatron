import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
from dataloader import DataLoaderForBert
from tqdm import tqdm
from galvatron.utils import set_seed, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.bert.hf_configs import bert_hf_configs, set_model_config
from galvatron.models.bert.arguments import model_args

def model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label):
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    prediction_scores, seq_relationship_score = outputs.prediction_logits, outputs.seq_relationship_logits
    loss_fct = nn.CrossEntropyLoss(ignore_index = -1)
    masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    loss = masked_lm_loss + next_sentence_loss
    return loss

def train(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")

    config = BertConfig.from_pretrained(bert_hf_configs(args.model_size))
    config = set_model_config(config, args, False)
    print(config)

    print("Creating Model...")
    model = BertForPreTraining(config)
    model.to(device)
    
    print("Creating Dataloader...")
    dataset = DataLoaderForBert(device)
    trainloader = DataLoader(
        dataset=dataset,
        batch_size=args.global_train_batch_size,
        shuffle=False
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profiler = GalvatronProfiler(args)
    profiler.set_profiler_single()

    profiler.profile_memory(0, "After creating model")
    print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            profiler.profile_time_start(iter)
            
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label = batch
            
            profiler.profile_memory(iter, "Before Forward")

            loss = model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label)

            profiler.profile_memory(iter, "After Forward")

            loss.backward()

            profiler.profile_memory(iter, "After Backward")
            
            optimizer.step()

            profiler.profile_memory(iter, "After optimizer_step")
            
            optimizer.zero_grad()

            print_loss(args, loss, ep, iter)
            
            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train')
    set_seed()
    train(args)