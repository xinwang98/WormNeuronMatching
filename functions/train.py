from functions.utils import *

def train(train_loader, model, criterion, optimizer, epoch, logger, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()


    for i, (patch_data,vertical_patch_data,pair_data, target) in enumerate(train_loader):
        if args.gpu is not None:
            patch_data = patch_data.cuda(args.gpu, non_blocking=True)
            pair_data = pair_data.cuda(args.gpu, non_blocking=True)
            vertical_patch_data = vertical_patch_data.cuda(args.gpu, non_blocking=True)

        # compute output

        output = model(patch_data,vertical_patch_data,pair_data)

        target = target.cuda(args.gpu, non_blocking=True)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), patch_data.size(0))
        top1.update(acc1[0], patch_data.size(0))
        top5.update(acc5[0], patch_data.size(0))

        # compute gradient and do SGD steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader),loss=losses, top1=top1, top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)

### val process###
def validate(val_loader, model, criterion, epoch, logger, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (patch_data,vertical_patch_data,pair_data, target) in enumerate(val_loader):

            if args.gpu is not None:
                patch_data = patch_data.cuda(args.gpu, non_blocking=True)
                pair_data = pair_data.cuda(args.gpu, non_blocking=True)
                vertical_patch_data = vertical_patch_data.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(patch_data, vertical_patch_data,pair_data)

            target = target.cuda(args.gpu, non_blocking=True)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), patch_data.size(0))
            top1.update(acc1[0], patch_data.size(0))
            top5.update(acc5[0], patch_data.size(0))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), loss=losses,top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
        set_tensorboard(log_dict, epoch, logger)
    return losses.avg,top1.avg.item()



