import os
import sys
import mmcv
sys.path.append('./')

from detect_utils.utils import load_config, make_optimizer, save_checkpoint
from models.loss import loss
from models.lpr_net import lpr_model
from dataset_utils.dataset_fatory import make_dataloader
def train_for_one_epoch(model, loss, train_loader, optimizer, epoch_number):
    model.train()
    model.cuda()
    #loss.train()
    for i, (images, targets) in enumerate(train_loader):
        #batch_size = images.size(0)
        images = images.cuda(async = True)
        targets = targets.cuda(async = True)
        preds = model(images)
        loss_value = loss(preds, targets)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%100 == 0:
            print('Epoch: [{epoch}] --Training Summary\t'
                  'Loss {loss_value:.2f}'.
                  format(epoch = epoch_number, loss_value = loss_value.item()))
    print('Epoch: [{epoch}] --Training Summary\t'
          'Loss {loss_value:.2f}'.format(epoch = epoch_number, loss_value = loss_value.item()))

def train():
    cfg_file = sys.argv[1]
    cfg = load_config(cfg_file)
    model = lpr_model(16)
    optimizer = make_optimizer(cfg, model)
    train_loader = make_dataloader(cfg)
    for epoch_number in range(cfg.epochs):
        train_for_one_epoch(model, loss, train_loader, optimizer, epoch_number)
        save_checkpoint({'state_dict': model.state_dict()}, dir_name = cfg.log_dir)
if __name__ == '__main__':
    train()
    

