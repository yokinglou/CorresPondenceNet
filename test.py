import os
import hydra
import torch
import logging
import omegaconf
import importlib
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CorresPondenceNet
from utils import ModelWrapper, geo_error_per_cp

logger = logging.getLogger(__name__)

def my_collect_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def test(cfg):
    log_dir = os.path.curdir

    # Load Dataset
    test_dataset = CorresPondenceNet(cfg, 'test')
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=cfg.batch_size, 
                                 shuffle=False, 
                                 num_workers=cfg.num_workers, 
                                 collate_fn=my_collect_fn)
    
    cfg.num_classes = cfg.embedding_size
    model_impl = getattr(importlib.import_module('.{}'.format(cfg.network.name), package='models'), '{}Model'.format(cfg.task.capitalize()))(cfg).cuda()
    model = ModelWrapper(model_impl).cuda()
    model.load_state_dict(torch.load('best.pth')["state_dict"])
    model.eval()

    pcds = []
    embeddings = []
    keypoints = []
    with torch.no_grad():
        for batch_data in test_dataloader:
            pcd = batch_data[0].detach().cpu().numpy()
            kp = batch_data[1].detach().cpu().numpy()
            outputs = model(pcd)
            embedding = outputs.detach().cpu().numpy()
            pcds.append(pcd)
            embeddings.append(embedding)
            keypoints.append(kp)

    pcds = np.concatenate(pcds)
    embeddings = np.concatenate(embeddings)
    keypoints = np.concatenate(keypoints)

    file_path = os.path.join(cfg.data_path, 'model_geodesic_mat/{}_geo_dists.h5'.format(test_dataset.name2id[cfg.class_name.capitalize()]))
    with h5py.File(file_path, 'r') as f:
        dist_mats = f['geo_dists'][:]
        mesh_names = f['mesh_names'][:].tolist()

    dist_mats_test_data = []
    for i in range(pcds.shape[0]):
        mesh = test_dataset.mesh_names[i]
        idx = mesh_names.index(mesh)
        dist_mats_test_data.append(dist_mats[idx])
    dist_mats_test_data = np.stack(dist_mats_test_data)

    errors = []
    for i in range(keypoints.shape[1]):
        kp_indices = keypoints[:, i]
        if cfg.dist_type == 'geodesic':
            error, valid = geo_error_per_cp(pcds, embeddings, kp_indices, dist_mats_test_data)
        elif cfg.dist_type == 'euclidean':
            error, valid = geo_error_per_cp(pcds, embeddings, kp_indices, None)
        else:
            raise ValueError
        if valid:
            logger.info('Geodesic Error of C{}: {:.5f}'.format(i, error))
            errors.append(error)
        else:
            logger.info('Invalid: C{} only has one point.'.format(i))

    logger.info('Category: {}'.format(cfg.class_name))
    logger.info('Valid Correspondences: {}/{}'.format(len(errors), keypoints.shape[-1]))
    logger.info('mean Geodesic Error: {:.5f}'.format(np.stack(errors).mean()))
    
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    cfg.log_path = 'log'
    logger.info(cfg.pretty())
    test(cfg)

if __name__ == "__main__":
    main()