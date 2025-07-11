from torch.utils.data import DataLoader, Subset

from misc.log_utils import log, dict_to_string
from datasets import wildtrack
from datasets import motdataset
from datasets import scout
from datasets.basedataset import MvSequenceSet
from datasets.utils import get_train_val_split_index


def get_scene_set(dataset_names, data_conf, training):
    sceneset_list = []

    if "wildtrack" in dataset_names:
        log.debug("Adding Wildtrack sequences to the dataset")
        sceneset_list.append(wildtrack.WildtrackSet(data_conf, training))

    # if "scout1" in dataset_names:
    #     log.debug("Adding SCOUT sequences to the dataset")
    #     sceneset_list.append(scout.ScoutSet(data_conf, training, "sync_frame_seq_1", single_cam=None))

    # if "scout1_mono" in dataset_names:
    #     log.debug("Adding SCOUT mono sequences to the dataset")
    #     for cam_name in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]:#[10, 13, 19, 24, 5, 23, 3, 2, 4]: #
    #         sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_1", single_cam=cam_name))

    if "scouttrain" in dataset_names:
        log.debug("Adding SCOUT train sequences to the dataset")
        sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_2", cam_name=[0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]))
        sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_1", cam_name=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]))

    if "scoutval" in dataset_names:
        log.debug("Adding SCOUT validation sequences to the dataset")
        sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_1", cam_name=[0, 1, 2, 3, 4, 5, 6, 7]))

    if "scoutmonotrain" in dataset_names:
        log.debug("Adding SCOUT mono train sequences to the dataset")
        for cam_name in [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
            sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_2", single_cam=cam_name))
        for cam_name in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
            sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_1", single_cam=cam_name))

    if "scoutmonoval" in dataset_names:
        log.debug("Adding SCOUT mono validation sequences to the dataset")
        for cam_name in [0, 1, 2, 3, 4, 5, 6, 7]:
            sceneset_list.append(scout.ScoutSet(data_conf, training, "sequence_1", single_cam=cam_name))
    
    #MOT20 Train
    if "mot20train" in dataset_names:
        for i in [1,2,3,5]:
            log.debug(f"Adding MOT20 train {i:02} to the datasets")
            sceneset_list.append(motdataset.MotDataset(data_conf, "MOT20", f"{i:02}", training, "train"))

    #MOT20 Test  
    if "mot20test" in dataset_names:
        for i in [4,6,7,8]:
            log.debug(f"Adding MOT20 test {i:02} to the datasets")
            sceneset_list.append(motdataset.MotDataset(data_conf, "MOT20", f"{i:02}", training, "test"))

    #MOT17 Train
    if "mot17train" in dataset_names:
        for i in [2,4,5,9,10,11,13]:
            log.debug(f"Adding MOT17 train {i:02} to the datasets")
            sceneset_list.append(motdataset.MotDataset(data_conf, "MOT17", f"{i:02}", training, "train"))

    #MOT17 Test  
    if "mot17test" in dataset_names:
        for i in [1,3,6,7,8,12,14]:
            log.debug(f"Adding MOT17 test {i:02} to the datasets")
            sceneset_list.append(motdataset.MotDataset(data_conf, "MOT17", f"{i:02}", training, "test"))

    return sceneset_list


def get_datasets(dataset_names, data_conf, training):
    sceneset_list = get_scene_set(dataset_names, data_conf, training)
    
    datasets = [MvSequenceSet(sceneset, data_conf) for sceneset in sceneset_list] 

    return datasets


def get_dataloader(data_conf):
    log.info(f"Building Datasets")
    log.debug(f"Data spec: {dict_to_string(data_conf)}")

    train_datasets = get_datasets(data_conf["train_datasets"], data_conf, True)
    train_datasets_val = get_datasets(data_conf["train_datasets"], data_conf, False)
    eval_only_datasets = get_datasets(data_conf["val_datasets"], data_conf, False)

    train_val_splits = [get_train_val_split_index(dataset, dataset_val, data_conf["split_proportion"]) for dataset, dataset_val in zip(train_datasets, train_datasets_val)]   

    train_dataloaders = list()
    val_dataloaders = list()

    for dataset, dataset_no_aug, train_val_split in zip(train_datasets, train_datasets_val, train_val_splits):
        #Add train dataset (possibly subset) to train dataloaders
        train_dataloaders.append(DataLoader(
            Subset(dataset, train_val_split[0]),
            shuffle=data_conf["shuffle_train"],
            batch_size=data_conf["batch_size"],
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            num_workers=data_conf["num_workers"]
            )
        )
        
        #Add split part of the train dataset to validation dataloaders
        if data_conf["split_proportion"] < 1:
            val_dataloaders.append(DataLoader(
                Subset(dataset_no_aug, train_val_split[1]),
                shuffle=False,
                batch_size=data_conf["batch_size"],
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                num_workers=data_conf["num_workers"]
                )
            )
    
    #add validation only dataset to val dataloaders
    for dataset in eval_only_datasets:
        val_dataloaders.append(DataLoader(
                Subset(dataset, list(range(len(dataset)))),
                shuffle=False,
                batch_size=data_conf["batch_size"],
                collate_fn=dataset.collate_fn,
                pin_memory=True,
                num_workers=data_conf["num_workers"]
            )
        )


    return train_dataloaders, val_dataloaders


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
def chunkify_dataloader(dataloader, chunk_size, overlap_size):
    log.debug(f"Original dataloader size: {len(dataloader.dataset)}")
    dataset = dataloader.dataset
    total_frames = sum(range(0, len(dataset), overlap_size))

    indices = []
    chunk_lengths = []
    for i in range(0, len(dataset), overlap_size):
        chunk_indices = range(i, min(i + chunk_size, len(dataset)))
        indices.extend(chunk_indices)
        chunk_lengths.append(len(chunk_indices))

    chunked_dataset = Subset(dataset, indices)
    
    chunked_dataloader = DataLoader(
        chunked_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory
    )
    
    log.debug(f"Chunked dataloader size: {len(chunked_dataloader.dataset)}")
    return chunked_dataloader, chunk_lengths

