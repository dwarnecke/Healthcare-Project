import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from project.utilities.confusion_matrix import ConfusionMatrix


def evaluate_performance(model, dataset, criterion):
    """
    Evaluate the performance of the given model on the given dataset.
    :param model: The model to evaluate
    :param dataset: The dataset to evaluate on
    :param criterion: The loss function to use
    :return: The model loss, accuracy, F1 score, and confusion matrix
    """

    device = torch.device('cuda')
    model = model.to(device)

    batch_size = 2048
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    samples_loaded = 0

    # Evaluate the model over a subset of the data
    model.eval()
    loss = 0
    confusion_matrix = ConfusionMatrix(model.classes)
    for x, y in dataloader:
        y = y.to(device, dtype=torch.int64)

        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
        confusion_matrix.add(y_pred, y)

        samples_loaded += batch_size
        if samples_loaded > 20000:
            break

        torch.cuda.empty_cache()

    # Calculate the performance metrics
    accuracy, recall, precision, f1 = confusion_matrix.calculate()

    return loss, accuracy, f1, confusion_matrix


def train_model(
        model: nn.Module,
        datasets: dict,
        learning_rate: float,
        batch_size: int,
        score: float):
    """
    Train the passed model over the given number of epochs.
    :param model: The model to train
    :param datasets: The training and validation datasets to use
    :param learning_rate: The learning rate to use
    :param batch_size: The batch size to use
    :param score: The f1 score to achieve to stop training
    :return: The trained model
    """

    device = torch.device('cuda')
    model = model.to(device)

    # Train the model over many epochs
    print(f"Training the model over many epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for epoch in range(128):

        # Update the model with many gradient steps
        model.train()
        dataloader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=12)
        for x, y in dataloader:
            y = y.to(device=device, dtype=torch.int64)
            y_pred = model(x)
            train_loss = criterion(y_pred, y)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Evaluate the model performance after each epoch
        model.eval()
        train_metrics = evaluate_performance(model, datasets['train'], criterion)
        train_loss, train_accuracy, train_f1, _ = train_metrics
        valid_metrics = evaluate_performance(model, datasets['valid'], criterion)
        valid_loss, valid_accuracy, valid_f1, _ = valid_metrics
        
        print(f"Epoch {epoch + 1} — ", end='')
        print(f"train/loss {train_loss:.4f} ; ", end='')
        print(f"train/acc {train_accuracy:.4f} ; ", end='')
        print(f"train/f1 {train_f1:.4f} ; ", end='')
        print(f"valid/loss {valid_loss:.4f} ; ", end='')
        print(f"valid/acc {valid_accuracy:.4f} ; ", end='')
        print(f"valid/f1 {valid_f1:.4f}")

        if valid_f1 > score: break

    return model



