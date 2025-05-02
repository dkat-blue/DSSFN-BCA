# src/engine.py
# Contains the training and evaluation loops for the DSSFN model.
# Supports different fusion mechanisms based on model output.
# <<< MODIFICATION: Changed import for config to relative import >>>

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score, confusion_matrix
import logging # Use logging

# Import config to check fusion mechanism using relative import
try:
    from . import config as cfg
except ImportError:
    # Provide default if run standalone or config fails
    class MockConfig:
        FUSION_MECHANISM = 'AdaptiveWeight' # Default fallback
    cfg = MockConfig()
    logging.warning("Could not import config in engine.py, using default FUSION_MECHANISM='AdaptiveWeight'")


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                device, epochs, loss_epsilon, use_scheduler=True):
    """
    Trains the DSSFN model, handling adaptive or cross-attention fusion.

    Args:
        model (nn.Module): The DSSFN model to train.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): The optimizer (e.g., AdamW).
        scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        epochs (int): The total number of epochs to train.
        loss_epsilon (float): Small value for stable fusion weight calculation (AdaptiveWeight only).
        use_scheduler (bool): Whether to step the learning rate scheduler.

    Returns:
        tuple: (model, history)
            - model (nn.Module): The trained model (best weights if val_loader provided).
            - history (dict): Dictionary containing training and validation loss/accuracy per epoch.
    """
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Determine fusion mechanism from the model instance if possible, else use config
    fusion_mechanism = getattr(model, 'fusion_mechanism', cfg.FUSION_MECHANISM)
    logging.info(f"Engine using Fusion Mechanism: {fusion_mechanism}")

    if train_loader is None:
        logging.error("Engine: train_loader is None. Cannot train.")
        return model, history
    if val_loader is None:
        logging.warning("Engine: val_loader is None. Cannot validate model during training.")

    for epoch in range(epochs):
        logging.info(f'\nEpoch {epoch+1}/{epochs}')
        logging.info('-' * 10)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass - output depends on fusion mechanism
            outputs = model(inputs)

            # --- Calculate Loss and Predictions (depends on fusion type) ---
            if fusion_mechanism == 'AdaptiveWeight':
                # Expects tuple (spec_logits, spatial_logits)
                if not isinstance(outputs, tuple) or len(outputs) != 2:
                     logging.error(f"Epoch {epoch+1}, Batch {batch_idx}: AdaptiveWeight fusion expected tuple of 2 logits, but got {type(outputs)}. Check model forward pass.")
                     # Handle error: maybe skip batch or use a default loss?
                     # For now, let's try to proceed assuming the first output is usable, but log error.
                     if isinstance(outputs, torch.Tensor):
                         combined_logits = outputs # Fallback: assume model returned single tensor incorrectly
                         loss = criterion(combined_logits, labels)
                         _, preds = torch.max(combined_logits, 1)
                     else:
                         continue # Skip batch if output is unusable
                else:
                    spec_logits, spatial_logits = outputs
                    # Calculate individual stream losses (no grad needed for weights)
                    with torch.no_grad():
                        loss_spec = criterion(spec_logits, labels)
                        loss_spat = criterion(spatial_logits, labels)
                        # Ensure losses are positive before taking reciprocal
                        alpha = 1.0 / (loss_spec.item() + loss_epsilon) if loss_spec.item() > 0 else 1.0 / loss_epsilon
                        beta = 1.0 / (loss_spat.item() + loss_epsilon) if loss_spat.item() > 0 else 1.0 / loss_epsilon
                        total_weight = alpha + beta
                        # Prevent division by zero or near-zero total_weight
                        alpha_norm = alpha / max(total_weight, loss_epsilon)
                        beta_norm = beta / max(total_weight, loss_epsilon)

                    # Combined logits using normalized weights
                    combined_logits = alpha_norm * spec_logits + beta_norm * spatial_logits
                    loss = criterion(combined_logits, labels) # Loss for backprop
                    _, preds = torch.max(combined_logits, 1)

            elif fusion_mechanism == 'CrossAttention':
                 # Expects single tensor (fused_logits)
                 if not isinstance(outputs, torch.Tensor):
                      logging.error(f"Epoch {epoch+1}, Batch {batch_idx}: CrossAttention fusion expected single tensor, but got {type(outputs)}. Check model forward pass.")
                      # Handle error: skip batch or use a default?
                      continue # Skip batch
                 else:
                    fused_logits = outputs
                    loss = criterion(fused_logits, labels)
                    _, preds = torch.max(fused_logits, 1)
            else:
                 raise ValueError(f"Unsupported fusion_mechanism in engine: {fusion_mechanism}")

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        # Avoid division by zero if total_samples is 0 (e.g., empty train_loader)
        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_train_acc = running_corrects.double() / total_samples if total_samples > 0 else 0.0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item() if isinstance(epoch_train_acc, torch.Tensor) else epoch_train_acc) # Ensure it's a float
        logging.info(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')

        # --- Validation Phase ---
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            running_val_corrects = 0
            total_val_samples = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    # Calculate Loss and Predictions (depends on fusion type)
                    if fusion_mechanism == 'AdaptiveWeight':
                        if not isinstance(outputs, tuple) or len(outputs) != 2:
                            logging.error(f"Epoch {epoch+1}, Val: AdaptiveWeight fusion expected tuple of 2 logits, but got {type(outputs)}. Check model forward pass.")
                            if isinstance(outputs, torch.Tensor):
                                combined_logits_val = outputs # Fallback
                                loss = criterion(combined_logits_val, labels)
                                _, preds = torch.max(combined_logits_val, 1)
                            else:
                                continue # Skip batch
                        else:
                            spec_logits, spatial_logits = outputs
                            loss_spec_val = criterion(spec_logits, labels)
                            loss_spat_val = criterion(spatial_logits, labels)
                            alpha_val = 1.0 / (loss_spec_val.item() + loss_epsilon) if loss_spec_val.item() > 0 else 1.0 / loss_epsilon
                            beta_val = 1.0 / (loss_spat_val.item() + loss_epsilon) if loss_spat_val.item() > 0 else 1.0 / loss_epsilon
                            total_weight_val = alpha_val + beta_val
                            alpha_norm_val = alpha_val / max(total_weight_val, loss_epsilon)
                            beta_norm_val = beta_val / max(total_weight_val, loss_epsilon)

                            combined_logits_val = alpha_norm_val * spec_logits + beta_norm_val * spatial_logits
                            loss = criterion(combined_logits_val, labels)
                            _, preds = torch.max(combined_logits_val, 1)

                    elif fusion_mechanism == 'CrossAttention':
                         if not isinstance(outputs, torch.Tensor):
                              logging.error(f"Epoch {epoch+1}, Val: CrossAttention fusion expected single tensor, but got {type(outputs)}. Check model forward pass.")
                              continue # Skip batch
                         else:
                            fused_logits = outputs
                            loss = criterion(fused_logits, labels)
                            _, preds = torch.max(fused_logits, 1)
                    else:
                        raise ValueError(f"Unsupported fusion_mechanism in engine: {fusion_mechanism}")

                    # Statistics
                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_corrects += torch.sum(preds == labels.data)
                    total_val_samples += inputs.size(0)

            # Avoid division by zero if total_val_samples is 0
            epoch_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0.0
            epoch_val_acc = running_val_corrects.double() / total_val_samples if total_val_samples > 0 else 0.0
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc.item() if isinstance(epoch_val_acc, torch.Tensor) else epoch_val_acc)
            logging.info(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info(f'*** Best validation accuracy updated: {best_val_acc:.4f} ***')
        else: # No validation loader
             history['val_loss'].append(None)
             history['val_acc'].append(None)

        # Step the Learning Rate Scheduler
        if use_scheduler and scheduler is not None:
            scheduler.step()
            # Handle potential deprecation of get_last_lr()
            try:
                current_lr = scheduler.get_last_lr()[0]
            except AttributeError: # Older PyTorch versions might use _last_lr
                 current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Current LR: {current_lr:.6f}")


    time_elapsed = time.time() - start_time
    logging.info(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    if val_loader:
        logging.info(f'Best val Acc: {best_val_acc:4f}')
        model.load_state_dict(best_model_wts) # Load best model weights
    else:
        logging.info('No validation performed, returning model from last epoch.')

    return model, history


def evaluate_model(model, test_loader, device, criterion, loss_epsilon):
    """
    Evaluates the trained DSSFN model on the test set, handling different fusion mechanisms.

    Args:
        model (nn.Module): The trained DSSFN model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device for evaluation ('cuda' or 'cpu').
        criterion (nn.Module): The loss function (used for adaptive fusion weights).
        loss_epsilon (float): Small value for stable fusion weight calculation (AdaptiveWeight only).

    Returns:
        tuple: (oa, aa, kappa, report, test_predictions, test_labels)
    """
    if test_loader is None:
        logging.error("Engine: test_loader is None. Cannot evaluate.")
        return 0.0, 0.0, 0.0, "No evaluation performed.", np.array([]), np.array([])

    # Determine fusion mechanism from the model instance if possible, else use config
    fusion_mechanism = getattr(model, 'fusion_mechanism', cfg.FUSION_MECHANISM)
    logging.info(f"\nStarting test set evaluation (Fusion: {fusion_mechanism})...")

    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # --- Get Predictions (depends on fusion type) ---
            if fusion_mechanism == 'AdaptiveWeight':
                if not isinstance(outputs, tuple) or len(outputs) != 2:
                    logging.error(f"Test Batch {batch_idx}: AdaptiveWeight fusion expected tuple of 2 logits, but got {type(outputs)}. Check model forward pass.")
                    if isinstance(outputs, torch.Tensor):
                        combined_logits_test = outputs # Fallback
                        _, preds = torch.max(combined_logits_test, 1)
                    else:
                        preds = torch.zeros_like(labels) # Assign dummy predictions
                else:
                    spec_logits, spatial_logits = outputs
                    try:
                        loss_spec_test = criterion(spec_logits, labels)
                        loss_spat_test = criterion(spatial_logits, labels)
                        alpha_test = 1.0 / (loss_spec_test.item() + loss_epsilon) if loss_spec_test.item() > 0 else 1.0 / loss_epsilon
                        beta_test = 1.0 / (loss_spat_test.item() + loss_epsilon) if loss_spat_test.item() > 0 else 1.0 / loss_epsilon
                        total_weight_test = alpha_test + beta_test

                        alpha_norm_test = alpha_test / max(total_weight_test, loss_epsilon)
                        beta_norm_test = beta_test / max(total_weight_test, loss_epsilon)

                        combined_logits_test = alpha_norm_test * spec_logits + beta_norm_test * spatial_logits
                        _, preds = torch.max(combined_logits_test, 1)

                    except Exception as e:
                        logging.error(f"Error calculating adaptive weights in test batch {batch_idx}: {e}")
                        logging.warning("Falling back to using only spatial logits for this batch.")
                        _, preds = torch.max(spatial_logits, 1) # Fallback

            elif fusion_mechanism == 'CrossAttention':
                 if not isinstance(outputs, torch.Tensor):
                      logging.error(f"Test Batch {batch_idx}: CrossAttention fusion expected single tensor, but got {type(outputs)}. Check model forward pass.")
                      preds = torch.zeros_like(labels) # Assign dummy predictions
                 else:
                    fused_logits = outputs
                    _, preds = torch.max(fused_logits, 1)
            else:
                raise ValueError(f"Unsupported fusion_mechanism in engine: {fusion_mechanism}")

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Evaluated batch {batch_idx+1}/{len(test_loader)}")

    end_time = time.time()
    logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")

    # Convert lists to numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # Calculate Metrics
    logging.info("\nCalculating metrics...")
    oa, aa, kappa, report = 0.0, 0.0, 0.0, "Metrics calculation failed."
    if len(all_labels_np) > 0 and len(all_preds_np) == len(all_labels_np):
        try:
            oa = accuracy_score(all_labels_np, all_preds_np)
            kappa = cohen_kappa_score(all_labels_np, all_preds_np)
            report = classification_report(all_labels_np, all_preds_np, digits=4, zero_division=0)
            cm = confusion_matrix(all_labels_np, all_preds_np)
            # Calculate AA (Average Accuracy per class) - handle potential division by zero
            class_accuracies = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1)!=0)
            aa = np.nanmean(class_accuracies[~np.isnan(class_accuracies)]) # Exclude NaN from mean calculation
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            oa, aa, kappa, report = 0.0, 0.0, 0.0, f"Error calculating metrics: {e}"
    else:
        logging.warning("Could not calculate metrics: Label or prediction arrays are empty or mismatched.")
        report = "Metrics calculation skipped due to empty/mismatched arrays."


    logging.info("\n--- Test Set Evaluation Results ---")
    logging.info(f"Overall Accuracy (OA): {oa:.4f}")
    logging.info(f"Average Accuracy (AA): {aa:.4f}")
    logging.info(f"Kappa Coefficient:      {kappa:.4f}")
    logging.info("\nClassification Report:")
    logging.info(report)
    # logging.info(f"Class Accuracies (Recalls): {class_accuracies}") # Optional detail

    return oa, aa, kappa, report, all_preds_np, all_labels_np
