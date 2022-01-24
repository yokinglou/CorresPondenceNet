import os
import sys
import hydra
import torch
import logging
import time
import omegaconf
import importlib
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CorresPondenceNet
from criterion import Loss
from utils import ModelWrapper
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def train(cfg):
    log_dir = os.path.curdir

    # Load Dataset
    train_dataset = CorresPondenceNet(cfg, 'train')
    val_dataset = CorresPondenceNet(cfg, 'val')
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True, 
                                  num_workers=cfg.num_workers, 
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.batch_size, 
                                num_workers=cfg.num_workers)
    
    cfg.num_classes = cfg.embedding_size
    model_impl = getattr(importlib.import_module('.{}'.format(cfg.network.name), package='models'), '{}Model'.format(cfg.task.capitalize()))(cfg).cuda()
    model = ModelWrapper(model_impl).cuda()

    writer = SummaryWriter('summaries')
    
    logger.info('Start training on CPNet...')

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=cfg.lr, 
                                 weight_decay=cfg.weight_decay)
    criterion = Loss(cfg)

    best_val_loss = sys.maxsize
    best_epoch = -1

    # Train one epoch
    def step():
        model.train()
        loss_total = 0.
        cnt = 0

        tic = time.time()
        for i, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            pcds = batch_data[0]
            kp_index = batch_data[1].cuda()
            mean_distance = batch_data[2]
            evecs = batch_data[-1].cuda()

            batch_size = cfg.batch_size

            outputs = model(pcds)

            embeddings = []
            labels = []
            inds = {}
            start_idx = 0
            end_idx = 0

            # traverse all models and find all embeddings of keypoints
            for i in range(batch_size):
                embedding_model = outputs[i] # Embedding of each model: [2048, 128]
                keypoints = kp_index[i] # Keypoint indexes: [78]

                for idx in range(len(keypoints)):
                    kp_idx = keypoints[idx] # the indexes of keypoints in a point cloud
                    if kp_idx < 0:
                        continue # not exist this keypoint
                    embedding_kp = embedding_model[kp_idx] # the corresponding embeddings of keypoints
                    embeddings.append(embedding_kp)
                    labels.append(idx)
                    end_idx += 1

                inds[i] = (start_idx, end_idx)
                start_idx = end_idx

            embeddings = torch.stack(embeddings) # [num_keypoints, k]
            labels = torch.tensor(labels).cuda() # [num_keypoints]

            loss_dict = criterion((embeddings, labels, pcds, outputs, mean_distance, evecs))
            loss = loss_dict['total']
            if torch.isnan(loss):
                logger.info("Exit with error: nan!")
                sys.exit()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            optimizer.step()
            loss_total += loss.data
            cnt += 1

        loss_avg = loss_total / cnt
        toc = time.time()
        
        writer.add_scalar('training-loss', loss_avg, epoch)
        logger.info("Epoch {} - Train loss: {:.5f} ({:.2f}s)".format(epoch, loss_avg, toc-tic))


    def evaluate(best_val_loss, best_epoch):
        model.eval()
        cnt = 0
        loss_val = 0
        
        for i, batch_data in enumerate(val_dataloader):
            pcds = batch_data[0]
            kp_index = batch_data[1].cuda()
            mean_distance = batch_data[2]
            evecs = batch_data[-1].cuda()
            batch_size = kp_index.shape[0]
            with torch.no_grad():
                outputs = model(pcds)

            embeddings = []
            labels = []
            inds = {}
            start_idx = 0
            end_idx = 0
            for i in range(batch_size):
                embedding_model = outputs[i]
                keypoints = kp_index[i]

                for idx in range(len(keypoints)):
                    kp_idx = keypoints[idx]
                    if kp_idx < 0:
                        continue

                    embedding_kp = embedding_model[kp_idx]
                    embeddings.append(embedding_kp)
                    labels.append(idx)
                    end_idx += 1

                inds[i] = (start_idx, end_idx)
                start_idx = end_idx

            embeddings = torch.stack(embeddings)
            labels = torch.tensor(labels).cuda()

            loss_dict = criterion((embeddings, labels, pcds, outputs, mean_distance, evecs))
            loss_val += loss_dict["total"]
            cnt += 1

        loss_avg = loss_val / cnt
        writer.add_scalar('val-loss', loss_avg, epoch)
        

        is_best = False
        if loss_avg < best_val_loss:
            best_val_loss = loss_avg
            is_best = True
            best_epoch = epoch

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, 'best.pth')

        logger.info("Val loss: {:.5f}. Best val loss: {:.5f} (epoch {})".format(loss_avg, best_val_loss, best_epoch))            
        return best_val_loss, best_epoch

    for epoch in range(cfg.max_epoch + 1):
        step()
        # Evaluation
        if epoch % cfg.eval_step == 0 and epoch > 0:
            best_val_loss, best_epoch = evaluate(best_val_loss, best_epoch)


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    cfg.log_path = 'log'
    logger.info(omegaconf.OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()