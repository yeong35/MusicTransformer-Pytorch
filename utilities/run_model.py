import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

import numpy as np # 확인용으로 넣어둠 이따 지우세요

import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy


# train_epoch
def train_epoch(cur_epoch, model, critic, classifier, dataloader, loss, classifier_loss, opt, critic_opt, classifier_opt, lr_scheduler=None, critic_lr_scheduler=None, classifier_lr_scheduler=None, args=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """
    GAN_mode = args.gan
    creative_mode = args.creative

    model.train()

    acc_nll_loss = 0
    acc_dis_loss = 0
    acc_gen_loss = 0
    acc_cla_loss = 0
    acc_cre_loss = 0

    acc_gan_accuracy = 0
    acc_class_accuracy = 0
    acc_creativity = 0

    acc_pitch_accuracy = 0

    critic_count = 0

    for batch_num, batch in enumerate(dataloader):

        total_loss = 0

        time_before = time.time()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())
        label = batch[2].to(get_device())

        y = model(x)        # (batch, sequence, vocab)

        acc_pitch_accuracy += float(compute_epiano_accuracy(y, tgt, interval = args.interval, octave = args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale))

        soft_y = F.gumbel_softmax(y, tau=1, hard=False)
        hard_y = F.gumbel_softmax(y, tau=1, hard=True)
        # print("y:",y.reshape(y.shape[0] * y.shape[1], -1).shape)
        # print("tgt:", tgt.flatten().shape)

        nll_loss = loss(y.reshape(y.shape[0] * y.shape[1], -1), tgt.flatten())  # 계산중 nan이 나오는 것을 확인함
        # print(x)
        # print(type(nll_loss),nll_loss)

        # if np.isnan(nll_loss.detach().cpu()):
        #     print(x)
        #     exit()
        #     torch.save(y,'pred.pt')
        #     torch.save(tgt, 'target.pt')
        #     print('error saved')
            


        total_loss += nll_loss

        if args.interval and args.octave:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_OCTAVE_INTERVAL).float()
        elif args.interval and not args.octave:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_INTERVAL).float()
        elif args.octave and args.fusion_encoding and args.absolute:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_OCTAVE_FUSION_ABSOLUTE).float()
        elif args.octave and args.fusion_encoding:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_OCTAVE_FUSION).float()
        elif not args.interval and args.octave:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_OCTAVE).float()
        elif args.interval and args.absolute and args.logscale:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE_RELATIVE).float()
        else:
            tgt = F.one_hot(tgt, num_classes=VOCAB_SIZE).float()


        if GAN_mode:
            D_fake = critic(hard_y)
            G_loss = - torch.mean(D_fake)
            total_loss += G_loss

        if creative_mode:
            generated_pred = classifier(hard_y)
            #creative_loss = classifier_loss(generated_pred, torch.ones(label.shape).to(get_device()) * 0.5)
            creative_loss = - (torch.log(generated_pred) + torch.log(1.0 - generated_pred)).mean()

            acc_cre_loss += float(creative_loss.detach().cpu())

            acc_creativity += 0.5 - torch.abs(generated_pred.detach().cpu() - 0.5).mean()

            acc_class_accuracy += ((generated_pred > 0.5) * label).float().mean()

            total_loss += creative_loss

        opt.zero_grad()

        if not torch.isnan(total_loss).any():
            total_loss.backward()

        opt.step()

        acc_nll_loss += float(nll_loss.detach().cpu())
        # print("T, nll_loss :", nll_loss.detach().cpu())
        # print("F, nll_loss :", float(nll_loss.detach().cpu()))
        # print("acc_loss :",acc_nll_loss)

        # discriminator update!

        if GAN_mode:

            y = model(x)

            hard_y = F.gumbel_softmax(y, tau=1, hard=True)

            #D_fake = critic(torch.argmax(y, -1).float())
            

            D_real = critic(tgt)
            D_fake = critic(hard_y)

            # During discriminator forward-backward-update
            gradient_penalty = _gradient_penalty(tgt, soft_y, critic)

            D_loss = - (torch.mean(D_real) - torch.mean(D_fake)) + gradient_penalty

            critic_opt.zero_grad()
            D_loss.backward()
            critic_opt.step()

            acc_dis_loss += float(D_loss.detach().cpu())
            acc_gen_loss += float(G_loss.detach().cpu())

            acc_gan_accuracy += ((D_real > 0.5).float().mean() + (D_fake < 0.5).float().mean()).detach().cpu() / 2

            if critic_lr_scheduler is not None:
                critic_lr_scheduler.step()

            critic_count += 1


        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.1)


        #for p in critic.parameters():
        #    p.data.clamp_(-0.01, 0.01)

        if lr_scheduler is not None:
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if critic_count == 0:
            critic_count = 1

        if((batch_num+1) % args.print_modulus == 0):
            print(SEPERATOR)
            print(f"Epoch {cur_epoch}, Batch {batch_num+1}/{len(dataloader)}")
            print(
                f"Generator LR: {get_lr(opt)}, Discriminator LR: {get_lr(critic_opt)}, Classifier LR: {get_lr(classifier_opt)}")
            print(f"Total Train loss: {float(total_loss):.5f}, NLL loss: {acc_nll_loss / (batch_num + 1):.5f}, Discriminator loss: {acc_dis_loss / (critic_count):.5f}, Generator loss: {acc_gen_loss / (critic_count):.5f}, Classifier loss: {acc_cla_loss / (critic_count):.5f}, Creative loss: {acc_cre_loss / (batch_num + 1):.5f}")
            print(f"Discriminator Accuracy: {float(acc_gan_accuracy) / (critic_count):.5f}, Classifier Accuracy: {float(acc_class_accuracy) / (critic_count):.5f}, Creativity: {float(acc_creativity) / (batch_num + 1):.5f}")
            print(f"Time (s): {time_took}")
            print(SEPERATOR)
            print("")

    return acc_nll_loss / len(dataloader), acc_pitch_accuracy / len(dataloader), acc_dis_loss / critic_count, acc_gen_loss / critic_count, acc_cre_loss / len(dataloader), float(acc_gan_accuracy) / critic_count, float(acc_class_accuracy) / critic_count, float(acc_creativity) / len(dataloader)

# eval_model
def eval_model(model, dataloader, loss, args):
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

            sum_acc += float(compute_epiano_accuracy(y, tgt, interval = args.interval, octave = args.octave, fusion=args.fusion_encoding, absolute=args.absolute, logscale=args.logscale))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc

def _gradient_penalty(real_data, generated_data, critic, gp_weight=10):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(get_device())

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(get_device())
    #if self.use_cuda:
    #    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(get_device()),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()