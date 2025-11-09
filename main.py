from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from Agents import FL
from config import *
from util import timefunc
from dataset import load_dataset, split_data_equal
import ModelConfig

import sys
import copy
import random
from datetime import datetime

from log import *


def fractionUsers(users, fraction: float):
    """
    Select a random fraction of the users
    """
    amount = max(1, round(fraction*len(users)))
    users = users[:]
    random.shuffle(users)
    return users[:amount]


class ClientCluster:
    """
    Simple helper that groups a handful of clients and reports a single aggregated update.
    """
    def __init__(self, members, cluster_id: int):
        if not members:
            raise ValueError("ClientCluster must contain at least one client")
        self.members = members
        self.cluster_id = cluster_id
        self.leader = members[0]
        # Reuse a scratch model to avoid repeated allocations while aggregating.
        self._aggregator = copy.deepcopy(self.leader.model)

    def __len__(self):
        return len(self.members)

    def localClusterUpdate(self):
        """
        Let every client in this cluster train locally and aggregate their models.
        """
        epoch = self.leader.account.getEpoch()
        modelBytes = self.leader.account.getModel()
        accumulator = self._aggregator
        accumulator.zero()
        total_size = 0
        for client in self.members:
            update = client.localUpdate(modelBytes=modelBytes, epoch=epoch, upload=False)
            if update["epoch"] != epoch:
                log.warning(f"Cluster {self.cluster_id}: skipping stale update from client {client.index}")
                continue
            if update["size"] == 0:
                log.warning(f"Cluster {self.cluster_id}: client {client.index} produced zero-sized update")
                continue
            total_size += update["size"]
            accumulator.federate_from_bytes(update["model"], update["size"])
        if total_size == 0:
            log.warning(f"Cluster {self.cluster_id}: no usable updates; skipping blockchain submission.")
            return None
        accumulator.divide(total_size)
        receipt = self.leader.account.localUpdate(epoch, total_size, accumulator.to_bytes())
        log.info(f"Cluster {self.cluster_id} committed aggregated update with {total_size} samples.")
        return receipt


def build_clusters(clients, cluster_size: int):
    """
    Group consecutive clients into clusters of at most cluster_size members.
    """
    if cluster_size <= 1:
        return []
    clusters = []
    for offset in range(0, len(clients), cluster_size):
        members = clients[offset:offset + cluster_size]
        if members:
            clusters.append(ClientCluster(members, len(clusters)))
    return clusters


def test_model(model, dataloader, means, stds):
    """
    Test the accuracy and loss of the model with given data loader.
    Mean and std values are used for data standardization.
    """
    model.eval()
    # testing
    test_loss = 0
    correct = 0
    for idx, (X, y) in enumerate(dataloader):
        sX = (X - means) / stds
        prediction = model(sX)
        # sum up batch loss
        test_loss += F.cross_entropy(prediction, y, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = prediction.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    
    return accuracy.item(), test_loss



@timefunc(log.info)
def main():
    global NUM_USERS, LossFunc
    global train_dataloaders, test_dataloader

    def preprocessStage(subset):
        log.info(f"Starting preprocess stage... {len(subset)} client(s) participate.")
        receipts = [client.localMeans() for client in subset]
        server.combineMeans(receipts)
        # Even though a subset participates in preprocessing,
        # all clients must get means and std
        for client in clients:
            client.getMeans()

        receipts = [client.localStds() for client in subset]
        server.combineStds(receipts)
        for client in clients:
            client.getStds()

    if PLATFORM_NAME == "eth":
        from EthPlatform import EthPlatform as Platform
        log.info("Using EthPlatform")
    elif PLATFORM_NAME == "dummy":
        from DummyPlatform import DummyPlatform as Platform
        log.info("Using DummyPlatform")
    else:
        raise ValueError(f"Unknown platform: {PLATFORM_NAME}")

    accounts = Platform.initAccounts(NUM_USERS)
    if NUM_USERS > len(accounts):
        NUM_USERS = len(accounts)
        log.error(f"Only {NUM_USERS} accounts are available in blockchain network. Limiting the number of users")

    log.info(f"Loading dataset: {DATASET_FILENAME}")

    dataset = load_dataset(DATASET_FILENAME, VALIDATION_SIZE)
    log.info(f"Training samples: {dataset.num_train_data}")
    log.info(f"Features: {dataset.num_features}")
    log.info(f"Labels: {dataset.num_labels}")

    test_dataloader = DataLoader(dataset.test, batch_size=TEST_BATCH_SIZE)
    train_dataloaders = split_data_equal(dataset.train, NUM_USERS, BATCH_SIZE)

    try:
        model = ModelConfig.__dict__[MODEL_NAME]
        log.info(f"Using model: {MODEL_NAME}")
    except KeyError:
        log.error(f"Model not found: {MODEL_NAME}")
        sys.exit(1)
    global_model = model(dataset.num_features, dataset.num_labels, MODEL_ARGS)
    global_model.to(INTERNAL_DTYPE.torch)
    local_model = copy.deepcopy(global_model)
    log.info(f"Byte size of the model: {len(local_model.to_bytes())}")
    # Global loss function
    FL.LossFunc = nn.CrossEntropyLoss()
    FL.Xfeatures = dataset.num_features

    # The last account is server
    server = FL.Server(accounts[-1], global_model)
    # First NUM_USERS accounts are clients.
    # Note that server is also a client when NUM_USERS == len(accounts)
    clients = [
            FL.Client(accounts[i], local_model, dataloader)
            for i, dataloader in enumerate(train_dataloaders)
            ]
    clusters = build_clusters(clients, CLUSTER_SIZE)
    if clusters:
        log.info(f"Clustered FL enabled: {len(clusters)} cluster(s) with up to {CLUSTER_SIZE} client(s) each.")

    if PREPROCESSING_FRACTION == 0.0:
        log.info(f"Fraction is 0; skipping preprocessing stage")
        server.skipPreprocess()
        for client in clients:
            client.getMeans()
            client.getStds()
    else:
        preprocessStage(fractionUsers(clients, PREPROCESSING_FRACTION))
    means = clients[0].means
    stds = clients[0].stds

    # Both on validation set
    accuracies = []
    losses = []

    using_clusters = len(clusters) > 0
    log.info("Starting training...")
    for i in tqdm(range(GLOBAL_EPOCHS)):
        if using_clusters:
            subset_clusters = fractionUsers(clusters, TRAINING_FRACTION)
            receipts = []
            for cluster in tqdm(subset_clusters):
                receipt = cluster.localClusterUpdate()
                if receipt is not None:
                    receipts.append(receipt)
        else:
            subset = fractionUsers(clients, TRAINING_FRACTION)
            receipts = [client.localUpdate() for client in tqdm(subset)]

        if not receipts:
            log.warning("No updates collected this round; skipping model averaging.")
            continue

        server.averageUpdates(receipts)
        if EVAL_PER_EPOCH:
            model = server.getModel()
            acc, loss = test_model(model, test_dataloader, means, stds)
            accuracies.append(acc)
            losses.append(loss)
            log.info(f"Accuracy on validation: {acc * 100.0:.4f}%")

    if EVAL_PER_EPOCH:
        add_results({
                "name": datetime.now().isoformat(),
                "accuracies": accuracies,
                "losses": losses,
            })
    else:
        model = server.getModel()
        acc, loss = test_model(model, test_dataloader, means, stds)
        log.info(f"Accuracy on validation: {acc * 100.0:.4f}%")


if __name__ == '__main__':
    main()
