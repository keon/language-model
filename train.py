import argparse
import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torchtext import data, datasets

from model import RNNModel


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=0.25,
                   help='initial learning rate')
    p.add_argument('-log_interval', type=int, default=100,
                   help='print log every _')
    p.add_argument('-save', type=str, default='./.save',
                   help='directory to save the trained weights')
    return p.parse_args()


def repackage_hidden(h):
    """Wraps hidden states to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, val_iter, vocab_size, use_cuda=True):
    model.eval()
    total_loss = 0

    hidden = model.init_hidden(val_iter.batch_size)
    for b, batch in enumerate(val_iter):
        x, y = batch.text, batch.target
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        output, hidden = model(x, hidden)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               y.contiguous().view(-1))
        total_loss += loss.data[0]
        hidden = repackage_hidden(hidden)
    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, vocab_size, grad_clip,
          log_interval, use_cuda=True):
    model.train()
    total_loss = 0

    hidden = model.init_hidden(train_iter.batch_size)
    for b, batch in enumerate(train_iter):
        x, y = batch.text, batch.target
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output, hidden = model(x, hidden)
        hidden = repackage_hidden(hidden)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               y.contiguous().view(-1))
        loss.backward()

        # Prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.data[0]
        if b % log_interval == 0 and b > 0:
            cur_loss = total_loss / log_interval
            print("[batch: %d] loss:%5.2f | pp:%5.2f" %
                  (b, cur_loss, math.exp(cur_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    use_cuda = torch.cuda.is_available()

    print("[!] preparing dataset...")
    TEXT = data.Field()
    train_data, val_data, test_data = datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_data, min_freq=10)
    train_iter, val_iter, test_iter = data.BPTTIterator.splits(
                (train_data, val_data, test_data),
                batch_size=args.batch_size, bptt_len=30, repeat=False)
    vocab_size = len(TEXT.vocab)
    print("[TRAIN]:%d\t[VALID]:%d\t[TEST]:%d\t[VOCAB]%d"
          % (len(train_iter), len(val_iter), len(test_iter), vocab_size))

    print("[!] Instantiating models...")
    model = RNNModel('LSTM', ntoken=vocab_size, ninp=600, nhid=600,
                     nlayers=2, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if use_cuda:
        model.cuda()
    print(model)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(model, optimizer, train_iter, vocab_size,
              args.grad_clip, args.log_interval, use_cuda)
        val_loss = evaluate(model, val_iter, vocab_size, use_cuda)
        print("[Epoch: %d] val-loss:%5.2f | val-pp:%5.2f" %
              (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model")
            if not os.path.isdir(args.save):
                os.makedirs(args.save)
            torch.save(model, './%s/lm_%d.pt' % (args.save, e))
            best_val_loss = val_loss
    test_loss = evaluate(model, test_iter, vocab_size, use_cuda)
    print("[Epoch: %d] test-loss:%5.2f | test-pp:%5.2f" %
          (e, test_loss, math.exp(test_loss)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP] - training stopped due to interrupt")
