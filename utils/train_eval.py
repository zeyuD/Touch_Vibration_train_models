import torch
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, precision_recall_curve
import numpy as np

def train_verification_model(model, train_loader, learning_rate=1e-4, max_grad_norm=1.0):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    for epoch in range(model.epochs):
        total, correct, loss_sum = 0, 0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch.float())
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            predicted = (out > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            loss_sum += loss.item()
        acc = correct * 100 / total if total > 0 else 0
        print(f"Epoch {epoch+1}: Loss = {loss_sum/len(train_loader):.6f}, Accuracy = {acc:.2f}%")

def evaluate_verification_model(model, test_loader, target_key):
    model.eval()
    correct = 0
    total = 0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            out = model(x_batch)
            predicted = (out > 0.5).float()
            all_outputs.extend(out.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    pred = (all_outputs > 0.5).astype(int)
    conf = confusion_matrix(all_labels, pred)
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
    ap_score = average_precision_score(all_labels, all_outputs)
    tn, fp, fn, tp = conf.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    acc = 100 * (tp + tn) / (tp + tn + fp + fn)
    print(f"[{target_key}] Accuracy: {acc:.2f}%, AUC: {auc_score:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")
    return {
        'target_key': target_key,
        'accuracy': acc,
        'auc': auc_score,
        'avg_precision': ap_score,
        'far': far,
        'frr': frr,
        'conf_matrix': conf,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'raw_scores': all_outputs,
        'raw_labels': all_labels 
    }
