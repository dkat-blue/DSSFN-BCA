# src/engine.py
# Contains functions for training and evaluating the DSSFN model.

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import logging
import copy # For deepcopying model state

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                device, epochs, loss_epsilon=1e-7, use_scheduler=True,
                save_best_model=True,
                early_stopping_enabled=False,
                early_stopping_patience=10,
                early_stopping_metric='val_loss', # 'val_loss' or 'val_accuracy'
                early_stopping_min_delta=0.0001):
    """
    Trains the model.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader, optional): DataLoader for validation data.
        device (torch.device): Device to train on ('cuda' or 'cpu').
        epochs (int): Number of epochs to train.
        loss_epsilon (float): Epsilon for AdaptiveWeight fusion loss stability.
        use_scheduler (bool): Whether to use the learning rate scheduler.
        save_best_model (bool): If True, saves the model state with the best validation accuracy.
        early_stopping_enabled (bool): If True, enables early stopping.
        early_stopping_patience (int): Number of epochs with no improvement to wait before stopping.
        early_stopping_metric (str): Metric to monitor for early stopping ('val_loss' or 'val_accuracy').
        early_stopping_min_delta (float): Minimum change in the monitored metric to qualify as an improvement.


    Returns:
        tuple: (trained_model, history)
            - trained_model (nn.Module): The trained model (best state if save_best_model is True).
            - history (dict): Dictionary containing training and validation loss/accuracy per epoch.
    """
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0
    best_model_state_dict = None

    # Early stopping specific variables
    epochs_no_improve = 0
    best_early_stopping_metric_val = float('inf') if early_stopping_metric == 'val_loss' else float('-inf')
    early_stopping_triggered = False

    if early_stopping_enabled and val_loader is None:
        logging.warning("Early stopping is enabled, but no validation loader is provided. Early stopping will be disabled.")
        early_stopping_enabled = False # Disable if no val_loader

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if model.fusion_mechanism == 'AdaptiveWeight':
                outputs, spec_loss, spat_loss = model(inputs)
                # Ensure spec_loss and spat_loss are scalars if they are tensors
                if isinstance(spec_loss, torch.Tensor): spec_loss = spec_loss.mean()
                if isinstance(spat_loss, torch.Tensor): spat_loss = spat_loss.mean()

                main_loss = criterion(outputs, labels)
                loss = main_loss + model.lambda_spec * spec_loss + model.lambda_spat * spat_loss
                # Add epsilon for stability if adaptive weights are very small
                loss = loss + loss_epsilon * (torch.exp(-model.log_var_spec) + torch.exp(-model.log_var_spat))

            else: # CrossAttention or other mechanisms not returning separate losses
                outputs = model(inputs)
                loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        log_message = f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}"

        if val_loader:
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                for inputs_val, labels_val in val_pbar:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                    if model.fusion_mechanism == 'AdaptiveWeight':
                        outputs_val, spec_loss_val, spat_loss_val = model(inputs_val)
                        if isinstance(spec_loss_val, torch.Tensor): spec_loss_val = spec_loss_val.mean()
                        if isinstance(spat_loss_val, torch.Tensor): spat_loss_val = spat_loss_val.mean()

                        main_loss_val = criterion(outputs_val, labels_val)
                        val_loss_epoch = main_loss_val + model.lambda_spec * spec_loss_val + model.lambda_spat * spat_loss_val
                        val_loss_epoch = val_loss_epoch + loss_epsilon * (torch.exp(-model.log_var_spec) + torch.exp(-model.log_var_spat))
                    else:
                        outputs_val = model(inputs_val)
                        val_loss_epoch = criterion(outputs_val, labels_val)


                    running_val_loss += val_loss_epoch.item() * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()
                    val_pbar.set_postfix({'val_loss': val_loss_epoch.item()})

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_accuracy = correct_val / total_val
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_accuracy)
            log_message += f", Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}"

            if save_best_model and epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                best_model_state_dict = copy.deepcopy(model.state_dict())
                log_message += " (New Best Val Acc!)"

            # Early stopping check
            if early_stopping_enabled:
                current_metric_val = epoch_val_loss if early_stopping_metric == 'val_loss' else epoch_val_accuracy
                improved = False
                if early_stopping_metric == 'val_loss':
                    if current_metric_val < best_early_stopping_metric_val - early_stopping_min_delta:
                        improved = True
                else: # val_accuracy
                    if current_metric_val > best_early_stopping_metric_val + early_stopping_min_delta:
                        improved = True

                if improved:
                    best_early_stopping_metric_val = current_metric_val
                    epochs_no_improve = 0
                    log_message += f" (ES metric improved to {current_metric_val:.4f})"
                else:
                    epochs_no_improve += 1
                    log_message += f" (ES patience: {epochs_no_improve}/{early_stopping_patience})"

                if epochs_no_improve >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in {early_stopping_metric} for {early_stopping_patience} epochs.")
                    early_stopping_triggered = True
                    # The loop will break after logging this epoch's results.
        logging.info(log_message)

        if use_scheduler and scheduler:
            scheduler.step()
            logging.info(f"LR Scheduler stepped. New LR: {scheduler.get_last_lr()[0]:.2e}")

        if early_stopping_triggered:
            break # Exit epoch loop

    if save_best_model and best_model_state_dict:
        logging.info(f"Loading best model state dict with Val Acc: {best_val_accuracy:.4f}")
        model.load_state_dict(best_model_state_dict)
    elif not best_model_state_dict and save_best_model and val_loader:
        logging.warning("save_best_model was True, but no best_model_state_dict was saved (possibly no improvement or val_loader issue).")


    return model, history


def evaluate_model(model, test_loader, device, criterion, loss_epsilon=1e-7):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.
        criterion (nn.Module): Loss function (can be None if test loss is not required).
        loss_epsilon (float): Epsilon for AdaptiveWeight fusion loss stability if calculating loss.

    Returns:
        tuple: (overall_accuracy, average_accuracy, kappa, report, all_preds, all_labels)
            - overall_accuracy (float)
            - average_accuracy (float)
            - kappa (float): Cohen's Kappa score.
            - report (str): Classification report.
            - all_preds (np.ndarray): Predictions for the test set.
            - all_labels (np.ndarray): True labels for the test set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_test_loss = 0.0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating Test Set", leave=False)
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            if model.fusion_mechanism == 'AdaptiveWeight' and criterion:
                outputs, spec_loss, spat_loss = model(inputs)
                if isinstance(spec_loss, torch.Tensor): spec_loss = spec_loss.mean()
                if isinstance(spat_loss, torch.Tensor): spat_loss = spat_loss.mean()
                main_loss = criterion(outputs, labels)
                loss = main_loss + model.lambda_spec * spec_loss + model.lambda_spat * spat_loss
                loss = loss + loss_epsilon * (torch.exp(-model.log_var_spec) + torch.exp(-model.log_var_spat))
                running_test_loss += loss.item() * inputs.size(0)
            elif criterion: # Other fusion or if criterion is provided
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
            else: # No criterion, just get outputs
                outputs = model(inputs)


            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0 or len(all_preds) == 0:
        logging.warning("Evaluation dataset was empty or no predictions were made.")
        return 0.0, 0.0, 0.0, "No data to evaluate.", np.array([]), np.array([])

    if criterion:
        test_loss = running_test_loss / len(test_loader.dataset)
        logging.info(f"Test Loss: {test_loss:.4f}")


    overall_accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, zero_division=0)

    # Calculate Average Accuracy (AA)
    class_accuracies = []
    for label_val in np.unique(all_labels): # Iterate over actual unique labels present
        # Check if class is in report_dict (it should be if present in all_labels)
        if str(label_val) in report_dict:
            class_accuracies.append(report_dict[str(label_val)]['recall']) # recall is accuracy for that class
        # else: # This case should ideally not happen if report_dict is comprehensive
            # class_accuracies.append(0.0) # Or handle as appropriate

    if class_accuracies:
        average_accuracy = np.mean(class_accuracies)
    else: # Should not happen if there are labels and predictions
        average_accuracy = 0.0


    logging.info(f"Overall Accuracy (OA): {overall_accuracy:.4f}")
    logging.info(f"Average Accuracy (AA): {average_accuracy:.4f}")
    logging.info(f"Kappa Score: {kappa:.4f}")
    # logging.info(f"Classification Report:\n{report_str}") # Usually too verbose for main log

    return overall_accuracy, average_accuracy, kappa, report_str, all_preds, all_labels
