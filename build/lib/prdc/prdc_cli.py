import argparse
import torch
from torch.utils.data import DataLoader
import tqdm
from .prdc import compute_prdc
from .Dataset import CustomDataset
from .Models import RandomX, TrainedX
from .utils import customtransforms

def main():
    parser = argparse.ArgumentParser(
                        prog = 'PRDC CLI',
                        description = 'This is a CLI for PRDC which is ready to use and it has support for vgg16 (R64) case of original paper, hoever if you wish you can choose more or less than 64 output features from vgg16',
                        epilog = '')

    parser.add_argument('-r', '--real_dir', type = str, help = 'Path to real images directory', required = True)
    parser.add_argument('-f', '--fake_dir', type = str, help = 'Path to fake images directory', required = True)
    parser.add_argument('-o', '--out_feats', type = int, help = 'Number of output features from vgg16', default = 64)
    parser.add_argument('-t', '--type', type = str, help = 'Use pretrained (T) or random (R) vgg16', default = 'R')
    parser.add_argument('-b', '--batch_size', type = int, help = 'Batch size for dataloader', default = 64)
    parser.add_argument('-n', '--num_workers', type = int, help = 'Number of workers for dataloader', default = 4)
    parser.add_argument('-d', '--device', type = str, help = 'Device to use for extracting embeddings from vgg16', default = 'cpu')
    parser.add_argument('-k', '--nearest_k', type = int, help = 'Number of nearst neighbors', default = 5)

    args = parser.parse_args()

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    out_feats = args.out_feats
    batch_size = args.batch_size
    num_workers = args.num_workers
    k = args.nearest_k
    m_type = args.type
    device = torch.device(args.device)


    real_dataset = CustomDataset(real_dir, transforms=customtransforms)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    fake_dataset = CustomDataset(fake_dir, transforms=customtransforms)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    assert m_type in ['R', 'T'], 'Model type must be R or T'
    if m_type == 'R':
        model = RandomX(out_feats=out_feats).to(device)
        model.eval()
    elif m_type == 'T':
        model = TrainedX(out_feats=out_feats).to(device)
        model.eval()

    batch_precision = []
    batch_recall = []
    batch_densities = []
    batch_coverages = []

    for real_batch, fake_batch in tqdm.tqdm(zip(real_loader, fake_loader)):
        real_batch = real_batch.to(device)
        fake_batch = fake_batch.to(device)

        real_embeddings = model(real_batch).cpu().numpy()
        fake_embeddings = model(fake_batch).cpu().numpy()

        metrics = compute_prdc(real_embeddings, fake_embeddings, nearest_k=k)

        batch_precision.append(metrics['precision'])
        batch_recall.append(metrics['recall'])
        batch_densities.append(metrics['density'])
        batch_coverages.append(metrics['coverage'])

    precision = sum(batch_precision) / len(batch_precision)
    recall = sum(batch_recall) / len(batch_recall)
    density = sum(batch_densities) / len(batch_densities)
    coverage = sum(batch_coverages) / len(batch_coverages)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Density: {density}")
    print(f"Coverage: {coverage}")
