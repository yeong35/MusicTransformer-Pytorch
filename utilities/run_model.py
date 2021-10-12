import torch
import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy


# train_epoch
def train_epoch(cur_epoch, model, critic, classifier, dataloader, loss, classifier_loss, opt, critic_opt, classifier_opt, lr_scheduler=None, critic_lr_scheduler=None, print_modulus=1):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    model.train()

    acc_nll_loss = 0
    acc_dis_loss = 0
    acc_gen_loss = 0
    acc_cla_loss = 0

    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())
        label = batch[2].to(get_device())

        y = model(x)

        
        D_fake = critic(torch.argmax(y, -1))
        
        # During discriminator forward-backward-update
        # D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
        
        # During generator forward-backward-update
        G_loss = - torch.mean(D_fake)
        
        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        flattened_tgt = tgt.flatten()
        
        nll_loss = loss.forward(y, flattened_tgt)
        
        total_loss = nll_loss + G_loss
        
        # generator update!
        if batch_num % 2 == 0:
            opt.zero_grad() 
            total_loss.backward()
            opt.step()
                
        # discriminator update!
    
        y = model(x)
        
        #D_fake = critic(torch.argmax(y, -1).float())
        D_real = critic(tgt)
        D_fake = critic(torch.argmax(y, -1))
        
        # During discriminator forward-backward-update
        D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
        
        critic_opt.zero_grad()
        
        D_loss.backward()
        critic_opt.step()

        # classifier update!
        classifier_pred = classifier(torch.argmax(y, -1))

        BCE_loss = classifier_loss(classifier_pred, label)

        classifier_opt.zero_grad()

        BCE_loss.backward()
        classifier_opt.step()



        acc_nll_loss += float(nll_loss)
        acc_dis_loss += float(D_loss)
        acc_gen_loss += float(G_loss)
        acc_cla_loss += float(BCE_loss)
        
        if critic_lr_scheduler is not None:
            critic_lr_scheduler.step()
        
        #for p in critic.parameters():
        #    p.data.clamp_(-0.01, 0.01)

        if lr_scheduler is not None:
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print(f"Epoch {cur_epoch}, Batch {batch_num+1}/{len(dataloader)}")
            print(f"LR: {get_lr(opt)}")
            print(f"Total Train loss: {float(total_loss):.5f}, NLL loss: {acc_nll_loss / (batch_num + 1):.5f}, Discriminator loss: {acc_dis_loss / (batch_num + 1):.5f}, Generator loss: {acc_gen_loss / (batch_num + 1):.5f}, Classifier loss: {acc_cla_loss / (batch_num + 1):.5f}")
            print("")
            print(f"Time (s): {time_took}")
            print(SEPERATOR)
            print("")

    return

# eval_model
def eval_model(model, dataloader, loss):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        for batch in dataloader:
            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y = model(x)

            sum_acc += float(compute_epiano_accuracy(y, tgt))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc
