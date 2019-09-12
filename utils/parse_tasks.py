import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from utils.train import test
from utils.SSTask import SSTask
from utils.SSHead import linear_on_layer3
from dset_classes.DsetNoLabel import DsetNoLabel

def parse_tasks(args, ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset):
    sstasks = []

    if args.rotation:
        print('Task: rotation prediction')
        from dset_classes.DsetSSRotRand import DsetSSRotRand

        digit = False
        if args.source in ['mnist', 'mnistm', 'svhn', 'svhn_exta', 'usps']:
            print('No rotation 180 for digits!')
            digit = True

        su_tr_dataset = DsetSSRotRand(DsetNoLabel(sc_tr_dataset), digit=digit)
        su_te_dataset = DsetSSRotRand(DsetNoLabel(sc_te_dataset), digit=digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSRotRand(DsetNoLabel(tg_tr_dataset), digit=digit)
        tu_te_dataset = DsetSSRotRand(DsetNoLabel(tg_te_dataset), digit=digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(4, args.width, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_rotation, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)

    if args.quadrant:
        print('Task: quadrant prediction')
        from dset_classes.DsetSSQuadRand import DsetSSQuadRand

        su_tr_dataset = DsetSSQuadRand(DsetNoLabel(sc_tr_dataset))
        su_te_dataset = DsetSSQuadRand(DsetNoLabel(sc_te_dataset))
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSQuadRand(DsetNoLabel(tg_tr_dataset))
        tu_te_dataset = DsetSSQuadRand(DsetNoLabel(tg_te_dataset))
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(4, args.width, 4).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_quadrant, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
    
    if args.flip:
        print('Task: flip prediction')
        from dset_classes.DsetSSFlipRand import DsetSSFlipRand

        digit = False
        su_tr_dataset = DsetSSFlipRand(DsetNoLabel(sc_tr_dataset), digit=digit)
        su_te_dataset = DsetSSFlipRand(DsetNoLabel(sc_te_dataset), digit=digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        tu_tr_dataset = DsetSSFlipRand(DsetNoLabel(tg_tr_dataset), digit=digit)
        tu_te_dataset = DsetSSFlipRand(DsetNoLabel(tg_te_dataset), digit=digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

        head = linear_on_layer3(2, args.width, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                                lr=args.lr_flip, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler,
                     su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
    return sstasks
