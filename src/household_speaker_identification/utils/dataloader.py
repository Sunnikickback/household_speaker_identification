import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.household_speaker_identification.utils.params import Params, DataParams


def get_utt_num_by_pos(ind, num, utt_count):
    return (np.sum(utt_count[:ind]) + num).astype(int)


def get_pos_by_utt_num(num, utt_count):
    ind = 0
    while num >= utt_count[ind]:
        num -= utt_count[ind]
        ind += 1
    return ind, num


def get_folder_name_by_ind(ind, mask, user_folders):
    return mask + str(ind) if user_folders is None else user_folders[ind]


def generate_households(data_params: DataParams):
    user_folders = os.listdir(data_params.DB_dir)
    #     print(f'User_folders size = {len(user_folders)}')

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
            #             print(f"member_id={member_id}")
            #             print(f"enrol_utts={enrol_utts}")
            #             print(f"utt_count[{member_id}]={utt_count[member_id]}")
            #             print(f"user_folders[{member_id}]={user_folders[member_id]}")

            used_utterances = np.append(used_utterances,
                                        np.apply_along_axis(lambda num: get_utt_num_by_pos(member_id, num, utt_count),
                                                            arr=enrol_utts, axis=0).astype(int)).astype(int)
            #             print(used_utterances)
            pos_utts = np.append(pos_utts, np.random.choice(np.delete(np.arange(utt_count[member_id]), enrol_utts),
                                                            data_params.evaluation_utt, replace=False))
        guests = np.sort(np.random.choice(np.delete(set_of_utts, used_utterances), data_params.guests_per_hh))
        pos_utts = np.apply_along_axis(lambda num: get_utt_num_by_pos(member_id, num, utt_count), arr=pos_utts, axis=0)
        guests_with_eval = np.append(guests, np.random.choice(pos_utts,
                                                              data_params.household_size -
                                                              data_params.guests_per_hh -
                                                              len(pos_utts)))
        households.append((used_utterances, guests, pos_utts))

    return households, utt_count, user_folders


class DataSetLoader(Dataset):
    def __init__(self, params: Params):
        super(DataSetLoader, self).__init__()
        self.params = params.data
        self.households, self.utt_count, self.user_folders = generate_households(self.params)
        self.member_nums = []
        self.data_list_emb1 = np.array([])
        self.data_list_emb2 = np.array([])

        for (household, guests, pos_utts) in self.households:
            emb1, emb2 = self.generate_pairs(household, guests, pos_utts)
            self.data_list_emb1 = np.append(self.data_list_emb1, emb1)
            self.data_list_emb2 = np.append(self.data_list_emb2, emb2)

        if self.params.random_batch:
            indices = np.arange(0, len(self.data_list_emb1))
            np.random.shuffle(indices)
            self.data_list_emb1 = self.data_list_emb1[indices]
            self.data_list_emb2 = self.data_list_emb2[indices]

    def generate_pairs(self, household, guests, pos_utts):
        num_members = len(household) // self.params.enrollment_utt
        self.member_nums.append(num_members)
        pos_emb1 = np.array([])
        pos_emb2 = np.array([])
        for i in range(num_members):
            tmp_emb1 = np.random.choice(np.arange(self.params.enrollment_utt), self.params.evaluation_utt)
            pos_emb1 = np.append(
                np.apply_along_axis(lambda pos: household[(i * self.params.enrollment_utt) + pos], arr=tmp_emb1,
                                    axis=0),
                pos_emb1)
            tmp_emb2 = np.arange(self.params.evaluation_utt)
            np.random.shuffle(tmp_emb2)
            pos_emb2 = np.append(
                np.apply_along_axis(lambda pos: pos_utts[(i * self.params.enrollment_utt) + pos], arr=tmp_emb2, axis=0),
                pos_emb2)
        neg_emb1 = np.random.choice(household, len(guests))
        neg_emb2 = np.random.choice(guests, len(guests), replace=False)
        emb1 = np.append(pos_emb1.astype(int), neg_emb1).astype(int)
        emb2 = np.append(pos_emb2.astype(int), neg_emb2).astype(int)

        np.random.shuffle(emb1)
        np.random.shuffle(emb2)
        return emb1, emb2

    def __getitem__(self, indices):
        emb1_list = np.array(self.data_list_emb1[indices].astype(int)).reshape(-1)
        emb2_list = np.array(self.data_list_emb2[indices].astype(int)).reshape(-1)

        ids1 = np.apply_along_axis(lambda x: get_pos_by_utt_num(x, self.utt_count)[0], arr=emb1_list, axis=0)
        ids2 = np.apply_along_axis(lambda x: get_pos_by_utt_num(x, self.utt_count)[0], arr=emb2_list, axis=0)
        emb1_list = np.apply_along_axis(lambda x: os.path.join(self.params.DB_dir,
                                                               self.user_folders[
                                                                   get_pos_by_utt_num(x, self.utt_count)[0]],
                                                               str(get_pos_by_utt_num(x, self.utt_count)[1][
                                                                       0])) + ".npy",
                                        arr=emb1_list, axis=0)
        emb2_list = np.apply_along_axis(lambda x: os.path.join(self.params.DB_dir,
                                                               self.user_folders[
                                                                   get_pos_by_utt_num(x, self.utt_count)[0]],
                                                               str(get_pos_by_utt_num(x, self.utt_count)[1][
                                                                       0])) + ".npy",
                                        arr=emb2_list, axis=0)

        emb1_list = np.array(emb1_list).reshape(-1)
        emb2_list = np.array(emb2_list).reshape(-1)
        print(emb1_list)
        emb1 = np.apply_along_axis(lambda x: np.load(x[0]), arr=emb1_list, axis=0)
        emb2 = np.apply_along_axis(lambda x: np.load(x[0]), arr=emb2_list, axis=0)

        labels = (ids1 == ids2).astype(int)

        return torch.FloatTensor(emb1), torch.FloatTensor(emb2), labels

    def __len__(self):
        return len(self.data_list_emb1)