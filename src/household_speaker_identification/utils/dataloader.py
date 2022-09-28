import os

import numpy as np
import torch
from torch.utils.data import Dataset

from household_speaker_identification.utils.params import Params, DataParams
from tqdm import trange


def get_utt_num_by_pos(ind, num, utt_count):
    return (np.sum(utt_count[:ind]) + num).astype(int)


def get_pos_by_utt_num(num, utt_count, only_id=False):
    ind = 0
    while ind < len(utt_count) and num >= utt_count[ind]:
        num -= utt_count[ind]
        ind += 1
    if only_id:
        return ind
    return ind, num


def get_folder_name_by_ind(ind, mask, user_folders):
    return mask + str(ind) if user_folders is None else user_folders[ind]


def generate_households(data_params: DataParams):
    user_folders = os.listdir(data_params.DB_dir)

    utt_count = np.array([len(os.listdir(os.path.join(data_params.DB_dir, id))) for id in user_folders])
    set_of_utts = np.arange(sum(utt_count)).astype(int)
    households = []
    num_unique_users = len(os.listdir(data_params.DB_dir))
    for memb_num in np.random.randint(data_params.min_hh_size, data_params.max_hh_size, data_params.hh_num):
        used_utterances = np.array([])
        pos_utts = np.array([])
        members = np.sort(np.random.choice(np.arange(num_unique_users), memb_num, replace=False))
        for member_id in members:
            assert utt_count[member_id] >= data_params.enrollment_utt
            assert utt_count[member_id] >= data_params.evaluation_utt + data_params.enrollment_utt
            enrol_utts = np.sort(np.random.choice(np.arange(utt_count[member_id]),
                                                  data_params.enrollment_utt, replace=False))

            used_utterances = np.append(used_utterances,
                                        np.array(list(map(lambda num: get_utt_num_by_pos(member_id, num, utt_count),
                                                          enrol_utts)))).astype(int)
            pos_utts_tmp = np.random.choice(np.delete(np.arange(utt_count[member_id]), enrol_utts),
                                                            data_params.evaluation_utt, replace=False).astype(int)
            pos_utts = np.append(pos_utts,
                                 np.array(list(map(lambda num:
                                                   get_utt_num_by_pos(member_id, num, utt_count), pos_utts_tmp))))

        guests = np.sort(np.random.choice(np.delete(set_of_utts, used_utterances), data_params.guests_per_hh))
        guests_with_eval = np.append(guests, np.random.choice(pos_utts,
                                                              data_params.household_size -
                                                              data_params.guests_per_hh -
                                                              len(pos_utts)))
        households.append((used_utterances, guests_with_eval, pos_utts, members))

    return households, utt_count, user_folders


class DataSetLoader(Dataset):
    def __init__(self, params: DataParams):
        super(DataSetLoader, self).__init__()
        self.params = params
        self.households, self.utt_count, self.user_folders = generate_households(self.params)
        self.member_nums = []
        self.data_list_emb1 = np.array([])
        self.data_list_emb2 = np.array([])
        self.data_list_labels = np.array([])

        if params.path_to_households is None or not os.path.exists(os.path.join(params.path_to_households, "emb1.npy")):
            for (household, guests, pos_utts, member_ids) in self.households:
                emb1, emb2, labels = self.generate_pairs(household, guests, pos_utts, member_ids)

                indices = np.arange(0, len(emb1))
                np.random.shuffle(indices)
                emb1 = emb1[indices]
                emb2 = emb2[indices]
                labels = labels[indices]

                self.data_list_emb1 = np.append(self.data_list_emb1, emb1)
                self.data_list_emb2 = np.append(self.data_list_emb2, emb2)
                self.data_list_labels = np.append(self.data_list_labels, labels)

            if params.path_to_households is not None:
                if not os.path.exists(params.path_to_households):
                    os.mkdir(params.path_to_households)
                np.save(os.path.join(params.path_to_households, "emb1.npy"), self.data_list_emb1)
                np.save(os.path.join(params.path_to_households, "emb2.npy"), self.data_list_emb2)
                np.save(os.path.join(params.path_to_households, "labels.npy"), self.data_list_labels)
        else:
            self.data_list_emb1 = np.load(os.path.join(params.path_to_households, "emb1.npy"))
            self.data_list_emb2 = np.load(os.path.join(params.path_to_households, "emb2.npy"))
            self.data_list_labels = np.load(os.path.join(params.path_to_households, "labels.npy"))

        if self.params.random_batch:
            indices = np.arange(0, len(self.data_list_emb1))
            np.random.shuffle(indices)
            self.data_list_emb1 = self.data_list_emb1[indices]
            self.data_list_emb2 = self.data_list_emb2[indices]
            self.data_list_labels = self.data_list_labels[indices]

        n_utt = sum(self.utt_count)
        if params.saved_data is None:
            print('Embedding loading started')
            self.all_data = list(map(lambda x:
                                     np.load(
                                         os.path.join(self.params.DB_dir, self.user_folders[
                                             get_pos_by_utt_num(x, self.utt_count, only_id=True)],
                                                      str(get_pos_by_utt_num(x, self.utt_count)[1])) + ".npy"),
                                     trange(n_utt)))
            np.save("../data/all_data_in_one.npy", self.all_data)
        else:
            self.all_data = np.load(params.saved_data)

    def generate_pairs(self, household, guests, pos_utts, member_ids):
        num_members = len(household) // self.params.enrollment_utt
        self.member_nums.append(num_members)
        pos_emb1 = np.array([])
        pos_emb2 = np.array([])
        for i in range(num_members):
            tmp_emb1 = np.random.choice(np.arange(self.params.enrollment_utt), self.params.evaluation_utt)
            pos_emb1 = np.append(pos_emb1,
                                 np.array(list(map(lambda pos: household[(i * self.params.enrollment_utt) + pos],
                                                   tmp_emb1))))
            tmp_emb2 = np.arange(self.params.evaluation_utt)
            np.random.shuffle(tmp_emb2)
            pos_emb2 = np.append(pos_emb2,
                                 np.array(list(map(lambda pos: pos_utts[(i * self.params.enrollment_utt) + pos],
                                                   tmp_emb2))))
        neg_emb1 = np.random.choice(household, len(guests))
        neg_emb2 = np.random.choice(guests, len(guests), replace=False)

        np.random.shuffle(neg_emb1)
        np.random.shuffle(neg_emb2)

        emb1 = np.append(pos_emb1.astype(int), neg_emb1).astype(int)
        emb2 = np.append(pos_emb2.astype(int), neg_emb2).astype(int)

        labels = np.array(list(map(lambda x, y:  self.generate_label(x, y, member_ids), emb1, emb2)))

        assert len(emb1) == len(emb2) == len(labels)
        return emb1, emb2, labels

    def generate_label(self, emb1_num, emb2_num, member_ids):
        emb1_id = get_pos_by_utt_num(emb1_num, self.utt_count, only_id=True)
        emb2_id = get_pos_by_utt_num(emb2_num, self.utt_count, only_id=True)
        if emb1_id == emb2_id:
            return 2
        elif emb2_id in member_ids:
            return 1
        elif emb1_id not in member_ids:
            print('Ti ebobo?????')
            raise Exception()
        else:
            return 0

    def __getitem__(self, indices):
        emb1_list = np.array(self.data_list_emb1[indices].astype(int)).reshape(-1)
        emb2_list = np.array(self.data_list_emb2[indices].astype(int)).reshape(-1)
        labels = np.array(self.data_list_labels[indices].astype(int)).reshape(-1)
        emb1 = self.all_data[emb1_list]
        emb2 = self.all_data[emb2_list]

        return torch.FloatTensor(emb1), torch.FloatTensor(emb2), labels

    def __len__(self):
        return len(self.data_list_emb1)
