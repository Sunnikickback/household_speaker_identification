import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from torch.utils.tensorboard import SummaryWriter
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


def train(params: Params, train_loader, eval_loader, writer):
    model = ScoringModel(params.scoring_model).cuda()

    optimizer = Adam(parameters=model.parameters(), lr=params.training.learning_rate)
    criterion = Loss()
    print("Train is starting")
    global_iter = 0
    global_val_iter = 0
    for epoch in range(params.training.epoch_num):
        print(f"{epoch} epoch started...")
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, writer, global_iter)
        global_iter = train_metrics['global_iter']

        writer.add_scalar("train_loss/epoch", train_metrics['epoch_loss'], epoch)
        writer.add_scalar("train_threshold/epoch", train_metrics['mean_thr'], epoch)
        writer.add_scalar("train_EER/epoch", train_metrics['mean_eer'], epoch)

        eval_metrics = evaluate_epoch(model, eval_loader, writer, global_val_iter)
        global_val_iter = eval_metrics['global_iter']
        # print("Time spent to compute global eer:", eval_metrics['time_spent'])

        writer.add_scalar("val_EER/epoch", eval_metrics['mean_eer'], epoch)
        writer.add_scalar("val_threshold/epoch", eval_metrics['mean_thr'], epoch)
        # writer.add_scalar("val_EER/epoch", eval_metrics['epoch_loss'], epoch)
        # writer.add_scalar("val_global_threshold/epoch", eval_metrics['global_threshold'], epoch)
        # writer.add_scalar("val_global_EER/epoch", eval_metrics['global_eer'], epoch)


def train_epoch(model, optimizer, criterion, train_loader, writer, global_iter):
    model.train()
    epoch_loss = 0.0
    iteration = 0
    thresholds = np.array([])
    eers = np.array([])
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
        global_iter += 1
        eer, threshold = compute_eer(positive_scores, negative_scores)

        writer.add_scalar("train_threshold/step", threshold, global_iter)
        writer.add_scalar("train_EER/step", eer, global_iter)
        writer.add_scalar("train_loss/step", loss, global_iter)

        eers = np.append(eers, eer)
        thresholds = np.append(thresholds, threshold)
        losses.append(loss.item())
        epoch_loss += (loss.item() - epoch_loss) / iteration
    print("epoch loss = ", epoch_loss)
    return {"mean_eer": np.mean(eers), "mean_thr": np.mean(thresholds),
            "global_iter": global_iter, "epoch_loss": epoch_loss}


def evaluate_epoch(model, eval_loader, writer, global_iter):
    model.eval()
    thresholds = np.array([])
    eers = np.array([])
    # global_scores = np.array([])
    # global_labels = np.array([])

    for data in eval_loader:
        emb1, emb2, labels = data
        batch_size = emb1.shape[0]
        emb1 = emb1.view(batch_size, -1).cuda()
        emb2 = emb2.view(batch_size, -1).cuda()
        labels = labels.cuda()
        scores = model(emb1, emb2)
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        eer, threshold = compute_eer(positive_scores, negative_scores)

        writer.add_scalar("val_EER/step", eer, global_iter)
        writer.add_scalar("val_threshold/step", threshold, global_iter)
        # global_scores = np.append(global_scores, scores)
        # global_labels = np.append(global_labels, labels)
        eers = np.append(eers, eer)
        thresholds = np.append(thresholds, threshold)
        global_iter += 1
    # print("Started computing global scores")
    # start_time = time.time()
    # positive_scores = global_scores[global_labels == 1]
    # negative_scores = global_scores[global_labels == 0]
    # global_eer, global_threshold = compute_eer(positive_scores, negative_scores)
    # end_time = time.time()
    return {"mean_eer": np.mean(eers), "mean_thr": np.mean(thresholds), "global_iter": global_iter}
            # "time_spent": (end_time-start_time)/1000, "global_eer": global_eer, "global_threshold": global_threshold}


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    with open(args.config_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    params = Params(**config)
    train_dataset = DataSetLoader(params.train_data)
    eval_dataset = DataSetLoader(params.eval_data)
    writer = SummaryWriter()
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.training.batch_size,
        num_workers=params.training.num_workers
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=params.training.batch_size,
        num_workers=params.training.num_workers
    )
    train(params, train_loader, eval_loader, writer)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
