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
        loss_epsilon (float): Epsilon for AdaptiveWeight fusion loss stability (for reciprocal calculation).
        use_scheduler (bool): Whether to use the learning rate scheduler.
        save_best_model (bool): If True, saves the model state with the best validation accuracy/loss.
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
    best_metric_val_for_saving_model = float('-inf') # Initialize for val_accuracy based saving
    if early_stopping_enabled and early_stopping_metric == 'val_loss':
        best_metric_val_for_saving_model = float('inf')


    best_model_state_dict = None

    # Early stopping specific variables
    epochs_no_improve = 0
    # Initialize based on whether we want to minimize loss or maximize accuracy
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
                # Model returns individual logits from each stream
                spec_logits, spat_logits = model(inputs)
                
                # Calculate loss for each stream separately
                loss_spec_stream = criterion(spec_logits, labels)
                loss_spat_stream = criterion(spat_logits, labels)

                # Calculate adaptive weights (alpha, beta) based on the reciprocal of stream losses
                # Using .detach() so these loss calculations for weights don't affect gradient flow for the losses themselves
                alpha = 1.0 / (loss_spec_stream.detach() + loss_epsilon)
                beta = 1.0 / (loss_spat_stream.detach() + loss_epsilon)
                
                # Normalize weights so they sum to 1
                sum_weights = alpha + beta
                alpha_norm = alpha / sum_weights
                beta_norm = beta / sum_weights
                
                # Combine logits using the adaptive weights
                final_combined_logits = alpha_norm * spec_logits + beta_norm * spat_logits
                
                # The main loss for backpropagation is calculated on these combined logits
                loss = criterion(final_combined_logits, labels)
                
                # For calculating training accuracy based on the combined decision
                outputs_for_metric = final_combined_logits

            else: # For 'CrossAttention' or other fusion mechanisms that return final logits directly
                outputs_for_metric = model(inputs) 
                loss = criterion(outputs_for_metric, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs_for_metric.data, 1)
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
            current_epoch_val_loss_for_metric = 0.0 # For early stopping based on val_loss
            current_epoch_val_acc_for_metric = 0.0  # For early stopping based on val_accuracy

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                for inputs_val, labels_val in val_pbar:
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                    if model.fusion_mechanism == 'AdaptiveWeight':
                        spec_logits_val, spat_logits_val = model(inputs_val)
                        
                        loss_spec_val_stream = criterion(spec_logits_val, labels_val)
                        loss_spat_val_stream = criterion(spat_logits_val, labels_val)

                        alpha_val = 1.0 / (loss_spec_val_stream.detach() + loss_epsilon)
                        beta_val = 1.0 / (loss_spat_val_stream.detach() + loss_epsilon)
                        
                        sum_weights_val = alpha_val + beta_val
                        alpha_norm_val = alpha_val / sum_weights_val
                        beta_norm_val = beta_val / sum_weights_val
                        
                        outputs_val = alpha_norm_val * spec_logits_val + beta_norm_val * spat_logits_val
                        
                        # Loss for this batch based on combined logits
                        val_loss_batch_item = criterion(outputs_val, labels_val).item()
                    else: # 'CrossAttention' or other
                        outputs_val = model(inputs_val)
                        val_loss_batch_item = criterion(outputs_val, labels_val).item()
                    
                    running_val_loss += val_loss_batch_item * inputs_val.size(0)
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()
                    val_pbar.set_postfix({'val_loss': val_loss_batch_item})

            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            epoch_val_accuracy = correct_val / total_val
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_accuracy)
            log_message += f", Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}"

            current_epoch_val_loss_for_metric = epoch_val_loss
            current_epoch_val_acc_for_metric = epoch_val_accuracy

            # Save best model based on validation accuracy (or loss if preferred)
            # This logic is independent of early stopping metric, but often they align
            if save_best_model:
                if early_stopping_metric == 'val_loss': # If saving based on best val_loss
                    if current_epoch_val_loss_for_metric < best_metric_val_for_saving_model:
                        best_metric_val_for_saving_model = current_epoch_val_loss_for_metric
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        log_message += " (New Best Val Loss for model save!)"
                else: # Default to saving based on best val_accuracy
                    if current_epoch_val_acc_for_metric > best_metric_val_for_saving_model:
                        best_metric_val_for_saving_model = current_epoch_val_acc_for_metric
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        log_message += " (New Best Val Acc for model save!)"


            # Early stopping check
            if early_stopping_enabled:
                current_metric_val_for_es = current_epoch_val_loss_for_metric if early_stopping_metric == 'val_loss' else current_epoch_val_acc_for_metric
                improved = False
                if early_stopping_metric == 'val_loss':
                    if current_metric_val_for_es < best_early_stopping_metric_val - early_stopping_min_delta:
                        improved = True
                else: # val_accuracy
                    if current_metric_val_for_es > best_early_stopping_metric_val + early_stopping_min_delta:
                        improved = True

                if improved:
                    best_early_stopping_metric_val = current_metric_val_for_es
                    epochs_no_improve = 0
                    log_message += f" (ES metric '{early_stopping_metric}' improved to {current_metric_val_for_es:.4f})"
                else:
                    epochs_no_improve += 1
                    log_message += f" (ES patience for '{early_stopping_metric}': {epochs_no_improve}/{early_stopping_patience})"

                if epochs_no_improve >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in {early_stopping_metric} for {early_stopping_patience} epochs.")
                    early_stopping_triggered = True
        
        logging.info(log_message)

        if use_scheduler and scheduler:
            scheduler.step()
            if scheduler.get_last_lr(): 
                logging.info(f"LR Scheduler stepped. New LR: {scheduler.get_last_lr()[0]:.2e}")

        if early_stopping_triggered:
            break # Exit epoch loop

    if save_best_model and best_model_state_dict:
        logging.info(f"Loading best model state dict (based on '{early_stopping_metric if val_loader else 'last epoch'}') with value: {best_metric_val_for_saving_model:.4f}")
        model.load_state_dict(best_model_state_dict)
    elif not best_model_state_dict and save_best_model and val_loader:
        logging.warning("save_best_model was True, but no best_model_state_dict was saved (possibly no improvement or val_loader issue). Final model is from last epoch.")
    
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
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_test_loss = 0.0
    has_criterion = criterion is not None

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating Test Set", leave=False)
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            if model.fusion_mechanism == 'AdaptiveWeight':
                spec_logits, spat_logits = model(inputs)
                
                if has_criterion:
                    loss_spec_stream_test = criterion(spec_logits, labels)
                    loss_spat_stream_test = criterion(spat_logits, labels)

                    alpha_test = 1.0 / (loss_spec_stream_test.detach() + loss_epsilon)
                    beta_test = 1.0 / (loss_spat_stream_test.detach() + loss_epsilon)
                    
                    sum_weights_test = alpha_test + beta_test
                    alpha_norm_test = alpha_test / sum_weights_test
                    beta_norm_test = beta_test / sum_weights_test
                    
                    outputs = alpha_norm_test * spec_logits + beta_norm_test * spat_logits
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item() * inputs.size(0)
                else: 
                    # If no criterion, cannot calculate alpha/beta based on current batch loss.
                    # For prediction, a simple average of logits can be used.
                    outputs = (spec_logits + spat_logits) / 2.0

            else: # 'CrossAttention' or other mechanisms
                outputs = model(inputs)
                if has_criterion:
                    loss = criterion(outputs, labels)
                    running_test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0 or len(all_preds) == 0:
        logging.warning("Evaluation dataset was empty or no predictions were made.")
        return 0.0, 0.0, 0.0, "No data to evaluate.", np.array([]), np.array([])

    if has_criterion and len(test_loader.dataset) > 0 :
        test_loss = running_test_loss / len(test_loader.dataset)
        logging.info(f"Test Loss: {test_loss:.4f}")
    elif has_criterion: # but dataset is empty
        logging.warning("Cannot compute test_loss, test_loader.dataset is empty.")


    overall_accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Using zero_division=0 to handle cases where a class might not have true samples or predictions,
    # which would otherwise raise a warning and affect report generation.
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, zero_division=0)

    # Calculate Average Accuracy (AA)
    class_accuracies = []
    unique_test_labels = np.unique(all_labels) # Get unique labels actually present in the test set

    for label_val_int in unique_test_labels: 
        label_val_str = str(label_val_int) # Keys in report_dict are strings
        if label_val_str in report_dict and 'recall' in report_dict[label_val_str]:
            class_accuracies.append(report_dict[label_val_str]['recall']) # Recall for a class is its accuracy
        # else: # This case means a label in unique_test_labels was not in report_dict (should be rare)
            # class_accuracies.append(0.0) # Or handle as appropriate, e.g. log a warning

    if class_accuracies: # Ensure not dividing by zero if class_accuracies is empty
        average_accuracy = np.mean(class_accuracies)
    else: # Should not happen if there are labels and predictions
        average_accuracy = 0.0
        if len(unique_test_labels) > 0 : # If there were labels but no recalls found
             logging.warning("Could not calculate AA: class_accuracies list is empty despite having unique labels.")


    logging.info(f"Overall Accuracy (OA): {overall_accuracy:.4f}")
    logging.info(f"Average Accuracy (AA): {average_accuracy:.4f}")
    logging.info(f"Kappa Score: {kappa:.4f}")
    # logging.info(f"Classification Report:\n{report_str}") # Usually too verbose for main log

    return overall_accuracy, average_accuracy, kappa, report_str, all_preds, all_labels
