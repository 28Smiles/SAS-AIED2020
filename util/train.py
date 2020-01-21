from .eval import evaluate
from transformers import *
from apex import amp
import os
import torch
from .tensorboard import *


def save_model(model_cpu, step, experiment):
    output_dir = 'logs/{}'.format(experiment)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model_cpu.state_dict(), os.path.join(output_dir, 'model_%d.torch' % step))


def train(model_cpu, model, optimizer, scheduler, train_dataset, val_datasets, tensorboard, experiment,
          train_batch_size, epochs, accumulation_steps):
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataset):
            outputs = model(
                batch[0].long().to('cuda'),
                attention_mask=batch[1].long().to('cuda'),
                labels=batch[2].long().to('cuda')
            )
            loss = outputs[0].mean()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            tensorboard.log_scalar('loss', loss.item(),
                                   step * train_batch_size + len(train_dataset) * epoch * train_batch_size)

            if (step + 1) % (len(train_dataset) / 2 / train_batch_size) == 0:
                print('Step {}/{}'.format(step, len(train_dataset)))

                save_model(model_cpu, step * train_batch_size + len(train_dataset) * epoch * train_batch_size,
                           experiment)
                model.eval()
                for key, val_dataset in val_datasets.items():
                    scores, _ = evaluate(model, val_dataset)
                    for k, s in map(lambda t: (key + '_' + t[0], t[1]), scores):
                        tensorboard.log_scalar(k, s,
                                               step * train_batch_size + len(train_dataset) * epoch * train_batch_size)

                model.train()


def training(train_dataset, val_datasets, model_cpu, experiment, learn_rate, warmup_steps, train_batch_size, epochs,
             accumulation_steps):
    tensorboard = Tensorboard('logs/%s' % experiment)
    model = model_cpu.to('cuda')

    model.zero_grad()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learn_rate, eps=1e-8)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=len(train_dataset) // accumulation_steps * epochs)

    train(model_cpu, model, optimizer, scheduler, train_dataset, val_datasets, tensorboard, experiment,
          train_batch_size, epochs, accumulation_steps)
