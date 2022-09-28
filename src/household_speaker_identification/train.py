import argparse

import torch
from torch.utils.data import DataLoader
import yaml

from torch.utils.tensorboard import SummaryWriter

from household_speaker_identification.models.cosine_model import CosineModel
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
    parser.add_argument("--model_type", type=str)
    return parser


def train(params: Params, model, train_loader, eval_loader, writer):

    optimizer = Adam(parameters=model.parameters(), lr=params.training.learning_rate)
    criterion = Loss()
    print("Train is starting")
    global_iter = 0
    for epoch in range(params.training.epoch_num):
        print(f"{epoch} epoch started...")
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, writer, global_iter)
        global_iter = train_metrics['global_iter']

        writer.add_scalar("train_loss/epoch", train_metrics['epoch_loss'], epoch)
        writer.add_scalar("train_threshold_known/epoch", train_metrics['thr_known'], epoch)
        writer.add_scalar("train_EER_known/epoch", train_metrics['eer_known'], epoch)
        writer.add_scalar("train_threshold_unknown/epoch", train_metrics['thr_unknown'], epoch)
        writer.add_scalar("train_EER_unknown/epoch", train_metrics['eer_unknown'], epoch)

        eval_metrics = eval(model, eval_loader)

        writer.add_scalar("val_EER_known/epoch", eval_metrics['eer_known'], epoch)
        writer.add_scalar("val_threshold_known/epoch", eval_metrics['thr_known'], epoch)
        writer.add_scalar("val_EER_unknown/epoch", eval_metrics['eer_unknown'], epoch)
        writer.add_scalar("val_threshold_unknown/epoch", eval_metrics['thr_unknown'], epoch)


def train_epoch(model, optimizer, criterion, train_loader, writer, global_iter):
    model.train()
    epoch_loss = 0.0
    iteration = 0

    global_scores = torch.tensor([])
    global_labels = torch.tensor([])

    losses = []
    for data in tqdm(train_loader):
        emb1, emb2, labels = data
        batch_size = emb1.shape[0]
        emb1 = emb1.view(batch_size, -1).cuda()
        emb2 = emb2.view(batch_size, -1).cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        scores = model(emb1, emb2)
        target_scores = scores[labels == 2]
        negative_scores = scores[labels != 2]
        loss = criterion(target_scores, negative_scores)
        loss.backward()
        optimizer.step()
        iteration += 1
        global_iter += 1

        writer.add_scalar("train_loss/step", loss, global_iter)

        global_labels = torch.cat([global_labels, labels.cpu().detach()])
        global_scores = torch.cat([global_scores, scores.cpu().detach()])
        losses.append(loss.item())
        epoch_loss += (loss.item() - epoch_loss) / iteration

    known_global_scores = global_scores[(global_labels == 2) | (global_labels == 1)]
    known_global_labels = global_labels[(global_labels == 2) | (global_labels == 1)]
    unknown_global_scores = global_scores[(global_labels == 2) | (global_labels == 0)]
    unknown_global_labels = global_labels[(global_labels == 2) | (global_labels == 0)]

    global_eer_known, global_threshold_known = compute_eer(known_global_labels, known_global_scores)
    global_eer_unknown, global_threshold_unknown = compute_eer(unknown_global_labels, unknown_global_scores)

    print("epoch loss = ", epoch_loss)
    return {"eer_known": global_eer_known, "thr_known": global_threshold_known,
            "eer_unknown": global_eer_unknown, "thr_unknown": global_threshold_unknown,
            "global_iter": global_iter, "epoch_loss": epoch_loss}


def eval(model, eval_loader):
    model.eval()
    global_scores = torch.tensor([])
    global_labels = torch.tensor([])
    for data in eval_loader:
        emb1, emb2, labels = data
        batch_size = emb1.shape[0]
        emb1 = emb1.view(batch_size, -1).cuda()
        emb2 = emb2.view(batch_size, -1).cuda()
        scores = model(emb1, emb2)
        global_labels = torch.cat([global_labels, labels])
        global_scores = torch.cat([global_scores, scores.cpu().detach()])

    known_global_scores = global_scores[(global_labels == 2) | (global_labels == 1)]
    known_global_labels = global_labels[(global_labels == 2) | (global_labels == 1)]
    unknown_global_scores = global_scores[(global_labels == 2) | (global_labels == 0)]
    unknown_global_labels = global_labels[(global_labels == 2) | (global_labels == 0)]

    global_eer_known, global_threshold_known = compute_eer(known_global_labels, known_global_scores)
    global_eer_unknown, global_threshold_unknown = compute_eer(unknown_global_labels, unknown_global_scores)

    return {"eer_known": global_eer_known, "thr_known": global_threshold_known,
            "eer_unknown": global_eer_unknown, "thr_unknown": global_threshold_unknown}


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    with open(args.config_yaml, "r") as stream:
        config = yaml.safe_load(stream)
    params = Params(**config)
    if args.model is "scoring_model":
        model = ScoringModel(params.scoring_model)
    else:
        model = CosineModel()

    model = model.cuda()

    writer = SummaryWriter()

    eval_dataset = DataSetLoader(params.eval_data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=params.training.batch_size,
        num_workers=params.training.num_workers,
        pin_memory=True
    )

    if not params.only_validate:
        train_dataset = DataSetLoader(params.train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params.training.batch_size,
            num_workers=params.training.num_workers,
            pin_memory=True
        )
        train(params, model, train_loader, eval_loader, writer)
    eval_metrics = eval(model, eval_loader)
    print("Eval metrics:")
    print(f"eer_known = {eval_metrics['eer_known']}; thr_known = {eval_metrics['thr_known']}")
    print(f"eer_unknown = {eval_metrics['eer_unknown']}; thr_unknown = {eval_metrics['thr_unknown']}")
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
