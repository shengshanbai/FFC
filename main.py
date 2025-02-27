import os
import argparse
import logging as logger
from resnet_def import create_net
from ffc_ddp import FFC
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from optim.optimizer import get_optim_scheduler
import torch.distributed as dist
import random
from torch.nn.parallel import DistributedDataParallel as ddp
import time
import torch_all
from torch_all.datasets import FaceRecDataset
from util.config_helper import load_config

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

dist_train = False


def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(id_loader, instance_loader, ffc_net, optimizer,
                    cur_epoch, conf, saved_dir, real_iter, scaler, lr_policy, lr_scheduler, warmup_epochs, max_epochs):
    """Tain one epoch by traditional training.
    """
    id_iter = iter(id_loader)
    random.seed(cur_epoch)
    avg_data_load_time = 0
    if dist_train:
        my_rank = dist.get_rank()
    else:
        my_rank = 0
    db_size = len(instance_loader)
    start_time = time.time()
    for batch_idx, instance_item in enumerate(instance_loader):
        ins_images = instance_item["images"]
        instance_label = instance_item["face_ids"]
        # Note that label lies at cpu not gpu !!!
        # start_time = time.time()
        if lr_policy != 'ReduceLROnPlateau':
            lr_scheduler.update(None, batch_idx * 1.0 / db_size)
        instance_images = ins_images.cuda(my_rank, non_blocking=True)
        try:
            pair_item = next(id_iter)
            images1, images2, id_indexes = pair_item["images_0"], pair_item["images_1"], pair_item["face_ids"]
        except:
            id_iter = iter(id_loader)
            pair_item = next(id_iter)
            images1, images2, id_indexes = pair_item["images_0"], pair_item["images_1"], pair_item["face_ids"]

        images1_gpu = images1.cuda(my_rank, non_blocking=True)
        images2_gpu = images2.cuda(my_rank, non_blocking=True)

        instance_images1, instance_images2 = torch.chunk(instance_images, 2)
        instance_label1, instance_label2 = torch.chunk(instance_label, 2)

        optimizer.zero_grad()
        x = torch.cat([images1_gpu, instance_images1])
        y = torch.cat([images2_gpu, instance_images2])
        x_label = torch.cat([id_indexes, instance_label1])
        y_label = torch.cat([id_indexes, instance_label2])

        with torch.cuda.amp.autocast():
            loss = ffc_net(x, y, x_label, y_label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        real_iter += 1
        if real_iter % 1000 == 0:
            loss_val = loss.item()
            lr = lr_scheduler.get_lr()[0]
            duration = time.time() - start_time
            left_time = (max_epochs * db_size - real_iter) * \
                (duration / 1000) / 3600
            logger.info('Iter %d Loss %.4f Epoch %d/%d Iter %d/%d Left %.2f hours' %
                        (real_iter, loss_val, cur_epoch, max_epochs, batch_idx + 1, db_size, left_time))
            if lr_policy == 'ReduceLROnPlateau':
                lr_scheduler.step(loss_val)
            start_time = time.time()

        if real_iter % 2000 == 0 and cur_epoch >= 10 and dist_train and dist.get_rank() == 0:
            snapshot_path = os.path.join(
                saved_dir, '%d.pt' % (real_iter // 2000))
            torch.save({'state_dict': ffc_net.module.probe_net.state_dict(), 'lru': ffc_net.module.lru.state_dict(
            ), 'fc': ffc_net.module.queue.cpu(), 'qp': ffc_net.module.queue_position_dict}, snapshot_path)

    return real_iter


def draw_instance_sample(dataset, count=20):
    import cv2
    pick_ids = random.choices(range(len(dataset)), k=count)
    for id in pick_ids:
        data_item = dataset[id]
        image_np = (data_item["image"]/0.0078125 +
                    127.5).numpy().transpose((1, 2, 0)).astype(np.uint8)
        face_id = data_item["face_id"]
        image_np = torch_all.utils.image_util.draw_lable(
            image_np, f"{face_id}")
        cv2.imwrite(f"./output/sample_{id}.jpg", image_np)


def draw_id_sample(dataset, count=20):
    import cv2
    pick_ids = random.choices(range(len(dataset)), k=count)
    for id in pick_ids:
        data_item = dataset[id]
        image_0 = (data_item["image_0"]/0.0078125 +
                   127.5).numpy().transpose((1, 2, 0)).astype(np.uint8)
        face_id = data_item["face_id"]
        image_0 = torch_all.utils.image_util.draw_lable(
            image_0, f"{face_id}")
        image_1 = (data_item["image_1"]/0.0078125 +
                   127.5).numpy().transpose((1, 2, 0)).astype(np.uint8)
        image_1 = torch_all.utils.image_util.draw_lable(
            image_1, f"{face_id}")
        cv2.imwrite(f"./output/sample_{face_id}_0.jpg", image_0)
        cv2.imwrite(f"./output/sample_{face_id}_1.jpg", image_1)


def train(conf):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if dist_train:
        conf.rank = int(os.environ["RANK"])
        conf.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        conf.device_id = int(os.environ["LOCAL_RANK"])
        conf.dist_backend = "nccl"
        torch.distributed.init_process_group(backend=conf.dist_backend)
        torch.distributed.barrier()
        conf.device = torch.device(conf.device, conf.device_id)
        torch.cuda.set_device(conf.device)
    else:
        conf.device_id=0
    instance_sampler = id_sampler = None
    instance_db = FaceRecDataset(
        conf.source_lmdb, pair=False)
    # draw_instance_sample(instance_db)
    # instance_db = MultiLMDBDataset(conf.source_lmdb, conf.source_file)
    if dist_train:
        instance_sampler = DistributedSampler(instance_db)
        shuffle = False
    else:
        instance_sampler = None
        shuffle = True
    instance_loader = DataLoader(instance_db,
                                 conf.batch_size,
                                 shuffle,
                                 instance_sampler,
                                 num_workers=4,
                                 pin_memory=False,
                                 drop_last=True,
                                 collate_fn=instance_db.train_collate_fn)
    id_db = FaceRecDataset(
        conf.source_lmdb, pair=True)
    # draw_id_sample(id_db)
    # id_db = PairLMDBDatasetV2(conf.source_lmdb, conf.source_file)
    if dist_train:
        id_sampler = DistributedSampler(id_db)
        shuffle = False
    else:
        id_sampler = None
        shuffle = True
    id_loader = DataLoader(id_db, conf.batch_size, shuffle, id_sampler,
                           num_workers=4, pin_memory=False, drop_last=True,
                           collate_fn=id_db.train_collate_fn)
    logger.info('#class %d' % len(id_db))
    net = FFC(conf.net_type, conf.feat_dim, conf.queue_size, conf.scale, conf.loss_type, conf.margin,
              conf.alpha, conf.neg_margin, conf.pretrained_model_path, len(id_db)).cuda(conf.device_id)

    if conf.sync_bn:
        sync_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    else:
        sync_net = net
    if dist_train:
        ffc_net = ddp(sync_net, [conf.device_id])
    else:
        ffc_net = sync_net
    # Only rank 0 has another validate process.
    optim_config = load_config('config/optim_config')
    optim, lr_scheduler = get_optim_scheduler(
        ffc_net.parameters(), optim_config)

    real_iter = 0
    logger.info('enter training procedure...')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(optim_config['epochs']):
        if optim_config['scheduler'] != 'ReduceLROnPlateau':
            lr_scheduler.update(epoch, 0.0)
        if instance_sampler is not None:
            instance_sampler.set_epoch(epoch)
        real_iter = train_one_epoch(id_loader, instance_loader, ffc_net, optim, epoch, conf, conf.saved_dir, real_iter,
                                    scaler, optim_config['scheduler'], lr_scheduler, optim_config['warmup'], optim_config['epochs'])

    # id_db.close()
    # instance_db.close()
    if dist_train:
        dist.destroy_process_group()


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='fast face classification.')
    conf.add_argument('--saved_dir', type=str,
                      default="./output/", help='snapshot directory')

    conf.add_argument('--net_type', type=str,
                      default='mobile', help='backbone type')
    conf.add_argument('--queue_size', type=int, default=7409,
                      help='size of the queue.')
    conf.add_argument('--print_freq', type=int, default=1000,
                      help='The print frequency for training state.')
    conf.add_argument('--pretrained_model_path', type=str, default='')
    conf.add_argument('--batch_size', type=int, default=12,
                      help='batch size over all gpus.')
    conf.add_argument('--alpha', type=float, default=0.99,
                      help='weight of moving_average')
    conf.add_argument('--loss_type', type=str, default='Arc',
                      choices=['Arc', 'AM', 'SV'], help="loss type, can be softmax, am or arc")
    conf.add_argument('--margin', type=float, default=0.5, help='loss margin ')
    conf.add_argument('--scale', type=float, default=32.0,
                      help='scaling parameter ')
    conf.add_argument('--neg_margin', type=float,
                      default=0.25, help='scaling parameter ')
    conf.add_argument('--sync_bn', action='store_true', default=False)
    conf.add_argument('--feat_dim', type=int, default=512,
                      help='feature dimension.')
    args = conf.parse_args()
    logger.info('Start optimization.')

    args.source_lmdb = '/home/ssbai/datas/glint360k_db'
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
