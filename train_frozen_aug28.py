import torch
import sys, os
sys.path.append('/home/jupyter-hannah/software/esm')
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data.dataloader import DataLoader
from esm.data import MissingBmrbDataset
from model_down_bigdata import FrozenSeq2Seq
print(torch.cuda.get_device_name(0))
import gc
import argparse
import numpy as np

from torch.optim.lr_scheduler import StepLR

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script with integer and string arguments')
    parser.add_argument('--epochs', type=int, default=1000, help='n_epochs')
    parser.add_argument('--mask_termini',action='store_true')
    parser.add_argument('--missing_class_wt', type=float, default=1.0, help='weight on missing class for cross entropy loss')
    parser.add_argument('--job_id', type=str, help='unique id to save models under')
    parser.add_argument('--train_set', type=str, default='train_set_22aug2024', help='training set (must exist in RelaxDB/split_files)')
    parser.add_argument('--valid_set', type=str, default='val_set_22aug2024', help='validation set (must exist in RelaxDB/split_files)')

    args = parser.parse_args()
    epochs = args.epochs
    batch_size=20
    missing_loss_weight = args.missing_class_wt

    dyn_train = MissingBmrbDataset(split=args.train_set, mask_termini = args.mask_termini, load_attn=True,
                                   root_path = os.path.expanduser('/home/jupyter-hannah/software/'))
    dyn_valid = MissingBmrbDataset(split=args.valid_set, root_path = os.path.expanduser('/home/jupyter-hannah/software/'),
                                  mask_termini = args.mask_termini, load_attn=True)

    train_loader = DataLoader(dataset=dyn_train,batch_size=batch_size,shuffle=True,
                        collate_fn=dyn_train.__collate_fn__,drop_last=True)
    valid_loader = DataLoader(dataset=dyn_valid,batch_size=batch_size,shuffle=False,
                        collate_fn=dyn_valid.__collate_fn__,drop_last=False)

    best_acc, best_acc_epoch=0,0
    gc.collect()
    torch.cuda.empty_cache()

    model = FrozenSeq2Seq().cuda()
    
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f'Trainable parameters: {trainable_params}')
    print(f'Non-trainable parameters: {non_trainable_params}')

    for epoch in range(epochs):
        print(f'epoch:{epoch + 1} start!')
        train_loss = 0
        train_acc = 0
        train_step = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            #seqs = torch.squeeze(batch['input_ids'])
            arrs = torch.as_tensor(batch['attn']).cuda()
            targets = torch.as_tensor(batch['targets']).cuda()
            outputs = model(arrs, targets=targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0]
            acc = loss_acc[1]['auroc']
            loss=torch.mean(loss)
            acc=torch.mean(acc)

            train_loss += loss
            train_acc += acc
            train_step += 1

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.8f}. Training AUROC: {:.8f}."
                      .format(train_step, len(train_loader), (train_loss / train_step),
                              (train_acc / train_step)))

        val_loss = 0
        val_acc = 0
        val_step = 0
        model.eval()
        for idx, batch in enumerate(valid_loader):
            arrs = torch.as_tensor(batch['attn']).cuda()
            targets = torch.as_tensor(batch['targets']).cuda()
            outputs = model(arrs, targets=targets)
            
            with torch.no_grad():
                outputs = model(arrs, targets=targets)
                loss_acc, value_prediction = outputs
                loss = loss_acc[0]
                acc = loss_acc[1]['auroc']
                
            loss=torch.mean(loss)
            acc=torch.mean(acc)
            val_loss += loss.item()
            val_acc += acc.item()
            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.8f}. Validating AUROC: {:.8f}.\n".
              format(val_step, len(valid_loader), (val_loss / val_step), (val_acc / val_step)))

        val_loss = val_loss / val_step
        val_acc = val_acc / val_step

        if val_acc > best_acc:
            save_data = {"model_state_dict": model.module.state_dict(),
                          "optim_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            print("Save model! Best val AUROC is: {:.8f}.".format(val_acc))
            torch.save(save_data, "best_model_%s.pt" % (args.job_id))
            best_acc = val_acc
        print(
            "\nEpoch: {} / {} finish. Training Loss: {:.8f}.  Validating Loss: {:.8f}.\n"
                .format(epoch + 1, epochs, train_loss / train_step, val_loss))
        
        scheduler.step()
