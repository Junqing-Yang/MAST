import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from tqdm import tqdm
from lib.MAST_Network import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import structure_loss,clip_gradient, adjust_lr
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
import math


def Parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40, help='epoch number')
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='training batch size')
    parser.add_argument('--train_size', type=int, default=352,
                        help='training dataset size')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.6, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5,
                        help='every n epochs decay learning rate')
    parser.add_argument(
        '--if_load_weight', type=bool, default=False, help='if use checkpoints')
    parser.add_argument(
        '--weight_path', type=str, default='./snapshot/', help='train from checkpoints')
    parser.add_argument('--if_val', type=bool,
                        default=False, help='if use Validation')
    parser.add_argument('--train_root', type=str, default='/data/SUN-SEG/TrainDataset',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/data/SUN-SEG/TestEasyDataset/Unseen',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default='./snapshot/train_pvtMAST_{}_lr_{}_log_{}/'.format(parser.parse_args().lr, datetime.now().strftime('%Y-%m-%d_%H-%M')),
                        help='the path to save model and log')
    parser.add_argument('--save_every', type=int, default=5,
                        help='How often to save a snapshot')
    return parser.parse_args()


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    destroy_process_group()


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        init_lr: int,
        decay_rate: int,
        decay_epoch: int,
        save_path: str,
        total_epoch: str,
        clip: int,
        writer,
        log,
        val_data,
        val_size,
        if_val,
        scheduler
    ) -> None:
        self.gpu_id = gpu_id
        model = model.to(self.gpu_id)
        self.model = DistributedDataParallel(model, device_ids=[
            gpu_id], find_unused_parameters=True)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        self.save_path = save_path
        self.writer = writer
        self.total_step = len(train_data)
        self.total_epoch = total_epoch
        self.clip = clip
        self.log = log
        self.val_data = val_data
        self.val_size = val_size
        self.if_val = if_val
        self.scheduler=scheduler

    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt

    def _run_batch(self, index, epoch, anchors, anchor_gts, references, reference_gts, average_loss):
        origin_shape = anchor_gts.shape
        anchor_gts = anchor_gts.view(-1, *origin_shape[2:])
        reference_gts = reference_gts.view(-1, *origin_shape[2:])
        anchor_preds, reference_preds = self.model(anchors, references)

        loss_anchor = structure_loss(anchor_preds[0], anchor_gts) + structure_loss(
            anchor_preds[1], anchor_gts) + structure_loss(anchor_preds[2], anchor_gts) + structure_loss(anchor_preds[3], anchor_gts)
        loss_reference = structure_loss(reference_preds[0], reference_gts) + structure_loss(
            reference_preds[1], reference_gts) + structure_loss(reference_preds[2], reference_gts) + structure_loss(reference_preds[3], reference_gts)
        loss = loss_anchor + loss_reference
        loss.backward()

        average_loss[0] += self.reduce_tensor(loss_anchor.data)
        average_loss[1] += self.reduce_tensor(loss_reference.data)


        clip_gradient(self.optimizer, self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if (index % (self.total_step//10) == 0 or index == self.total_step or index == 1):
            print('[Train Info]: {} , GPU [{:02d}] , Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}, Loss1: {:.4f}, Loss2: {:0.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.gpu_id, epoch, self.total_epoch, index, self.total_step, loss.data, loss_anchor.data, loss_reference.data))
            if self.gpu_id == 0:
                self.writer.add_scalars('GPU0_Step_Loss',
                        {'Loss_anchor': loss_anchor.data, 'Loss_reference': loss_reference.data,
                            'Loss_total': loss.data},
                        global_step=index+(epoch-1)*self.total_step)
            

        if index == self.total_step:
            if self.gpu_id == 0:
                average_loss[0] = average_loss[0] / self.total_step
                average_loss[1] = average_loss[1] / self.total_step

                self.writer.add_scalars('Epoch_AverageLoss',
                                        {'Loss_anchor': average_loss[0], 'Loss_reference': average_loss[1],
                                            'Loss_total': sum(average_loss)},
                                        global_step=epoch)
                self.writer.add_scalar('Epoch_Learning_Rate',
                                       self.optimizer.state_dict()[
                                           'param_groups'][0]['lr'],
                                       global_step=epoch)

                print("[Train Info]: [GPU{:02d}] Epoch {:03d} | Loss_AVG: {:.4f} | lr: {}".format(
                    self.gpu_id, epoch, sum(average_loss), self.optimizer.state_dict()['param_groups'][0]['lr']))
            

        return loss.item()

    def _run_epoch(self, epoch, now_best_loss):

        self.train_data.sampler.set_epoch(epoch)
        train_epoch_loss = 0

        b_sz = len(next(iter(self.train_data))[0])

        print("[Train Info]: [GPU{:02d}] Epoch {:03d} | Batchsize: {:02d} | Steps: {:03d}".format(
            self.gpu_id, epoch, b_sz, len(self.train_data)))
    
        average_loss = [0,0]
        for i, (anchors, anchor_gts, references, reference_gts) in tqdm(enumerate(self.train_data, start=1), total=len(self.train_data), desc="train"):
            anchors, anchor_gts, references, reference_gts = anchors.to(self.gpu_id), anchor_gts.to(
                self.gpu_id), references.to(self.gpu_id), reference_gts.to(self.gpu_id)
            train_epoch_loss += self._run_batch(
                i, epoch, anchors, anchor_gts, references, reference_gts, average_loss)

        train_epoch_loss=train_epoch_loss/len(self.train_data)

        adjust_lr(self.optimizer, self.init_lr, epoch,
                  self.decay_rate, self.decay_epoch)


        if train_epoch_loss < now_best_loss[0]:
            dist.barrier()
            if self.gpu_id == 0:
                now_best_loss[0] = train_epoch_loss
                print(
                    "train_epoch_loss: {:.4f} | now_best_loss: {:.4f}".format(
                        train_epoch_loss, now_best_loss[0]
                    )
                )
                self._save_checkpoint(epoch=epoch, if_best=True)

        if self.gpu_id == 0 and train_epoch_loss < now_best_loss[0]:
            now_best_loss[0]=train_epoch_loss
            print("train_epoch_loss: {:.4f} | now_best_loss: {:.4f}".format(
               train_epoch_loss, now_best_loss[0]))
            self.log.append("train_epoch_loss: {:.4f} | now_best_loss: {:.4f}\n".format(
                train_epoch_loss, now_best_loss[0]))
            self._save_checkpoint(epoch=epoch, if_best=True)

        if self.gpu_id == 0:
            # self.writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
            self.writer.add_scalar(
                'Loss-Epoch', train_epoch_loss, global_step=epoch)

    def _save_checkpoint(self, epoch, if_best):
        checkpoint = self.model.module.state_dict()
        prefix = ""
        if if_best is True:
            prefix = "best_"

        torch.save(checkpoint, os.path.join(self.save_path,
                   '{}MAST_checkpoint_epoch_{:02d}.pth'.format(prefix, epoch)))

        print("Epoch {:02d} | Training checkpoint saved at {}MAST_checkpoint_epoch_{:02d}.pth".format(
            epoch, prefix, epoch))

    def train(self, max_epochs: int):
        try:
            best_epoch,best_mae  = 1,1000
            now_best_loss = [100000]
            self.model.train()
            for epoch in range(1, max_epochs+1):
                self._run_epoch(epoch, now_best_loss)
                if epoch % self.save_every == 0:
                    dist.barrier()
                    if self.gpu_id == 0:
                        self._save_checkpoint(epoch, False)
                if self.if_val and self.gpu_id == 0:
                    best_epoch,best_mae =self.val(epoch, best_epoch, best_mae)
                    self.model.train()

        except KeyboardInterrupt:
            if self.gpu_id == 0:
                print('Keyboard Interrupt: save model and exit.')
                cleanup()
            raise

    def val(self, epoch, best_epoch, best_mae):
        try:
            self.model.eval()
            with torch.no_grad():
                mae_sum = 0
                for i in tqdm(range(min(self.val_size,1000)), desc="valid"):
                    anchor_image, anchor_gt, name_anchor, _ = self.val_data.load_data()
                    reference_image, reference_gt, name_reference, _ = self.val_data.load_data()

                    anchor_gt = np.asarray(anchor_gt, np.float32)
                    anchor_gt /= (anchor_gt.max() + 1e-8)
                    anchor_image = anchor_image.to(self.gpu_id)

                    reference_gt = np.asarray(reference_gt, np.float32)
                    reference_gt /= (reference_gt.max() + 1e-8)
                    reference_image = reference_image.to(self.gpu_id)

                    anchor_pres, reference_pres = self.model(
                        anchor_image, reference_image)

                    anchor_pres = F.interpolate(anchor_pres[3], size=anchor_gt.shape,
                                             mode='bilinear', align_corners=False)
                    reference_pres = F.interpolate(
                        reference_pres[3], size=anchor_gt.shape, mode='bilinear', align_corners=False)

                    anchor_pres = anchor_pres.sigmoid().data.cpu().numpy().squeeze()
                    anchor_pres = (anchor_pres - anchor_pres.min()) / \
                        (anchor_pres.max() - anchor_pres.min() + 1e-8)

                    reference_pres = reference_pres.sigmoid().data.cpu().numpy().squeeze()
                    reference_pres = (reference_pres - reference_pres.min()) / \
                        (reference_pres.max() - reference_pres.min() + 1e-8)

                    mae_sum += np.sum(np.abs(anchor_pres - anchor_gt)) * 1.0 / \
                        (anchor_gt.shape[0] * anchor_gt.shape[1])
                    try:
                        mae_sum += np.sum(np.abs(reference_pres - reference_gt)) * \
                            1.0 / (reference_gt.shape[0]
                                   * reference_gt.shape[1])
                    except:
                        pass
                mae = mae_sum / self.val_size
            
                if self.gpu_id == 0:
                    self.writer.add_scalar(
                        'Valid_MAE', mae, global_step=epoch)

                if epoch == 1:
                    best_mae = mae
                else:
                    if mae < best_mae:
                        best_mae = mae
                        best_epoch = epoch
                        if self.gpu_id == 0:
                            self._save_checkpoint(epoch, True)

                print('[Val Info]: [GPU{:02d}] Epoch {:03d} | MAE: {:.4f} | bestEpoch: {:03d} | bestMAE: {:.4f}'.format(self.gpu_id,epoch, mae, best_epoch, best_mae))
                
                return best_epoch,best_mae

        except KeyboardInterrupt:
            if self.gpu_id == 0:
                print('Keyboard Interrupt: save model and exit.')
                self._save_checkpoint(epoch, False)
                print('Save checkpoints after Keyboard Interrupt successfully!')
                cleanup()
            raise

def load_val_objs(rank, args, log):
    if rank == 0:
        print("load validation data...")
        # log.append("load validation data\n")

    val_data = test_dataset(image_root=args.val_root + '/Frame/',
                            gt_root=args.val_root + '/GT/',
                            testsize=args.train_size)

    return val_data, len(val_data)


def load_train_objs(rank, args, log):
    # load data
    if rank == 0:
        print('load training data...')


    train_loader = get_loader(image_root=args.train_root,
                              batchsize=args.batch_size,
                              trainsize=args.train_size,
                              num_workers=0)

    # build the model
    model = Network(channel=32)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    if args.if_load_weight:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(
            args.weight_path, map_location=map_location,weights_only=True))
        if rank == 0:
            print('load model from ', args.weight_path.split("/")[-1])

    return train_loader, model, optimizer,None


def main_train(rank: int, world_size: int, args, log):
    writer = SummaryWriter(os.path.join(args.save_path, 'summary'))
    ddp_setup(rank, world_size)

    train_data, model, optimizer,scheduler= load_train_objs(rank, args, log)
    val_data = None
    val_size = 0

    if rank==0:
        if args.if_val:
            val_data, val_size = load_val_objs(rank, args, log)
        
        trainer = Trainer(model, train_data, optimizer, rank, args.save_every,
                        args.lr, args.decay_rate, args.decay_epoch, args.save_path, args.epoch, args.clip, writer, log, val_data, val_size, args.if_val,scheduler)
    else:
        trainer = Trainer(model, train_data, optimizer, rank, args.save_every,
                        args.lr, args.decay_rate, args.decay_epoch, args.save_path, args.epoch, args.clip, writer, log, None, 0, False,scheduler)

    trainer.train(args.epoch)
    writer.close()

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = Parse()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    world_size = torch.cuda.device_count()
    log = []

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Start training...")
    mp.spawn(main_train, args=(world_size, args, log), nprocs=world_size)
    print("End training...")
