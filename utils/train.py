import torch
from utils.get_mmd import get_mmd

def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return 1-correct/total

def train(args, net, ext, sstasks, criterion_cls, optimizer_cls, scheduler_cls,
            sc_tr_loader, sc_te_loader, tg_te_loader):
    net.train()
    for sstask in sstasks:
        sstask.head.train()
        sstask.scheduler.step()

    epoch_stats = []
    for batch_idx, (sc_tr_inputs, sc_tr_labels) in enumerate(sc_tr_loader):
        for sstask in sstasks:
            sstask.train_batch()

        sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
        optimizer_cls.zero_grad()
        outputs_cls = net(sc_tr_inputs)
        loss_cls = criterion_cls(outputs_cls, sc_tr_labels)
        loss_cls.backward()
        optimizer_cls.step()


        if batch_idx % args.num_batches_per_test == 0:
            sc_te_err = test(sc_te_loader, net)
            tg_te_err = test(tg_te_loader, net)
            mmd = get_mmd(sc_te_loader, tg_te_loader, ext)

            us_te_err_av = []
            for sstask in sstasks:
                err_av, err_sc, err_tg = sstask.test()
                us_te_err_av.append(err_av)
            
            epoch_stats.append((batch_idx, len(sc_tr_loader), mmd, tg_te_err, sc_te_err, us_te_err_av))
            display = ('Iteration %d/%d:' %(batch_idx, len(sc_tr_loader))).ljust(24)
            display += '%.2f\t%.2f\t\t%.2f\t\t' %(mmd, tg_te_err*100, sc_te_err*100)
            for err in us_te_err_av:
                display += '%.2f\t' %(err*100)
            print(display)
    return epoch_stats
