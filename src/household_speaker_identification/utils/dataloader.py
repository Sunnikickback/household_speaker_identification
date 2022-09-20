import os

import numpy as np
import torch


def get_utt_num_by_pos(ind, num, utt_count):
    return np.sum(utt_count[:ind]) + num


def get_pos_by_utt_num(num, utt_count):
    ind = 0
    while num >= utt_count[ind]:
        num -= utt_count[ind]
        ind += 1
    return ind, num


def get_folder_name_by_ind(ind, mask, user_folders):
    return mask + str(ind) if user_folders is None else user_folders[ind]


def generate_households(DB_dir: str, min_hh_size=2, max_hh_size=8, hh_num=1000, unique_users=1251, guests_per_hh=250,
                        enrollment_utt=4, evaluation_utt=10, user_folders=None, mask="id1"):
    utt_count = np.array([len(os.listdir(os.path.join(DB_dir, id))) for id in os.listdir(DB_dir)])
    set_of_utts = set(range(sum(utt_count)))
    households = []
    for memb_num in np.random.randint(min_hh_size, max_hh_size, hh_num):
        used_utterances = np.array([])
        members = np.random.choice(np.arange(unique_users), memb_num, replace=False)
        for member_id in members:
            folder = get_folder_name_by_ind(member_id, mask, user_folders)
            assert utt_count[member_id] >= enrollment_utt
            if utt_count[member_id] <= evaluation_utt + enrollment_utt:
                Warning(f"Lack of speaker {folder} utterances")
            utts = np.random.choice(np.arange(utt_count[member_id]),
                                    min(enrollment_utt + evaluation_utt, utt_count[member_id]), replace=False)
            used_utterances = np.append(used_utterances,
                                        list(map(lambda num: get_utt_num_by_pos(member_id, num, utt_count), utts)))

        guests = np.random.choice(list(set_of_utts - set(used_utterances)), guests_per_hh, replace=False)
        households.append(np.array(list(set(used_utterances).union(guests)), dtype=np.int32))

    return households, utt_count



