import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from household_speaker_identification.models.scoring_model import ScoringModel
from household_speaker_identification.utils.dataloader import DataSetLoader
from household_speaker_identification.utils.loss import Loss
from household_speaker_identification.utils.metrics import compute_eer
from household_speaker_identification.utils.optimizers import Adam
from household_speaker_identification.utils.params import Params
from tqdm import tqdm

def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str)
    return parser


def train(params: Params, train_loader, eval_loader):
    model = ScoringModel(params.scoring_model).cuda()

    optimizer = Adam(parameters=model.parameters(), lr=1e-2)
    criterion = Loss()
    print("Train is starting")
    for epoch in range(params.training.epoch_num):
        print(f"{epoch} epoch started...")
        train_metrics = train_epoch(model, optimizer, criterion, train_loader)
        eval_metrics = evaluate_epoch(model, eval_loader)


def train_epoch(model, optimizer, criterion, train_loader):
    model.train()
    epoch_loss = 0.0
    iteration = 0
    thresholds = []
    eers = []
    losses = []
    for data in tqdm(train_loader):
        emb1, emb2, labels = data
        batch_size = emb1.shape[0]
        emb1 = emb1.view(batch_size, -1).cuda()
        emb2 = emb2.view(batch_size, -1).cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        scores = model(emb1, emb2)
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        loss = criterion(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()
        iteration += 1
        eer, threshold = compute_eer(positive_scores, negative_scores)
        eers.append(eer)
        thresholds.append(threshold)
        losses.append(loss.item())
        epoch_loss += (loss.item() - epoch_loss) / iteration
    print("epoch loss = ", epoch_loss)
    return {"losses": epoch_loss, "eers": eers, "thresholds": thresholds, "epoch_loss": epoch_loss}


def evaluate_epoch(model, eval_loader):
    model.eval()
    eers = []
    thresholds = []
    for data in eval_loader:
        emb1, emb2, labels = data
        emb1 = emb1.cuda()
        emb2 = emb2.cuda()
        labels = labels.cuda()
        scores = model(emb1, emb2)
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        eer, threshold = compute_eer(positive_scores, negative_scores)
        eers.append(eer)
        thresholds.append(threshold)
    return eers, thresholds


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    with open(args.config_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    params = Params(**config)
    train_dataset = DataSetLoader(params.train_data)
    eval_dataset = DataSetLoader(params.eval_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.training.batch_size,
        num_workers=32
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=params.training.batch_size,
        num_workers=32
    )
    train(params, train_loader, eval_loader)

if __name__ == '__main__':
    main()