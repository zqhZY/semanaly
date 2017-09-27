import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def fit(train, val, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    if args.cuda:
        model.cuda()
    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        for batch in train:

            feature, target = batch.text, batch.labels
            # print target

            target.data.sub_(1)
            x = feature.data.numpy()
            x = x.T
            feature = autograd.Variable(torch.from_numpy(x))
            # print feature[0]
            # print type(feature)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()

            out = model(feature)
            # print out

            # print out
            # print target.view(-1)
            loss = F.cross_entropy(out, target.view(-1))
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 1 == 0:
                # print torch.max(out, 1)[1].view(target.size()).data
                corrects = (torch.max(out, 1)[1].view(target.size()).data == target.data).sum()
                #print torch.max(out, 1)[1].view(target.size()).data
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % 1000 == 0:
                current_accuracy = eval(val, model, args)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    if not os.path.isdir(args.save_dir):
                        os.makedirs(args.save_dir)
                    print("save model at step %s, accuracy is %d".format(str(steps), current_accuracy))
                    save_path = os.path.join(args.save_dir, "snapshot.mdl")
                    torch.save(model, save_path)
    return model


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.labels
        target.data.sub_(1)  # batch first, index align

        x = feature.data.numpy()
        x = x.T
        feature = autograd.Variable(torch.from_numpy(x))

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0] / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                            size))
    return accuracy, avg_loss

