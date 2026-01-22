import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import segmentation_models_pytorch as smp

# For multi-class segmentation
# mode can be 'binary', 'multiclass' or 'multilabel'
loss_fn = smp.losses.FocalLoss(mode='multiclass', alpha=0.0004593042226938101, gamma=2.0)
def train(net, positive_class_weight, lr, epochs, trainloader, valloader):
    best_model_path = "best_model.pt"
    record = 0
    net = net.cuda()
    print(net)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., positive_class_weight]).cuda())
    #criterion = smp.losses.FocalLoss(mode='multiclass', gamma=2.0, alpha=0.25)
    #criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([300.]))
    #criterion = nn.MSELoss()
    # there's some instability in training, and gradient of hard examples
    # should there be sth like a focal loss? the imperfect data annotation can cause lots of issues though
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    record = 0.0
    for epoch in range(epochs):
        total_loss, num_batches = 0.0, 0.0
        net.train()
        for i, (x, y) in tqdm(enumerate(trainloader)):
            #print(Counter(np.array(y).flatten()))
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y.squeeze(1).to(torch.long))
            probs = torch.softmax(outputs, dim=1)[:,1,:,:]
            dice = 2 * (probs * y).sum() / (probs.sum() + y.sum())
            (loss).backward()
            optimizer.step()
            total_loss += (loss).item()
            #print(loss.item())
            num_batches += 1
        print(f"Epoch {epoch} train loss: {total_loss / num_batches}")
        train_losses.append(total_loss / num_batches)
        total_loss, num_batches = 0.0, 0.0
        net.eval()
        with torch.no_grad():
            all_pred = []
            all_gt = []
            all_probs = []
            total_loss = 0.0
            num_batches = 0
            
            # Process valloader in one pass
            for i, (x, y) in tqdm(enumerate(valloader)):
                x, y = x.cuda(), y.cuda()
                outputs = net(x)
                loss = criterion(outputs, y.squeeze(1).to(torch.long))
                probs = torch.softmax(outputs, dim=1)[:,1:,:,:]
                dice = 2 * (probs * y).sum() / (probs.sum() + y.sum())
                #(loss + dice)
                #total_loss += loss.item()
                total_loss += (loss).item()
                num_batches += 1
                
                # Get probabilities and predictions for this batch
                batch_probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
                batch_pred = torch.argmax(outputs, dim=1)
                
                # Process each sample in batch
                for j in range(x.shape[0]):
                    # Check if image is empty (replicates val_ds check)
                    if x[j].sum() == 0:
                        print("Empty image")
                        print(f"Batch {i}, Sample {j}")
                        continue
                    
                    # Get probability (max over spatial dimensions)
                    prob = batch_probs[j].max().item()
                    all_probs.append(prob)
                    
                    # Get prediction (whether any positive pixel exists)
                    output_mask = batch_pred[j]
                    lbl_mask = y[j, 0]  # Assuming y has shape [batch, 1, H, W]
                    
                    pred_label = int(output_mask.sum() != 0)
                    true_label = int(lbl_mask.sum() != 0)
                    
                    all_pred.append(pred_label)
                    all_gt.append(true_label)
            
            # Calculate validation loss
            val_loss = total_loss / num_batches
            print(f"Epoch {epoch} val loss: {val_loss}")
            val_losses.append(val_loss)
            
            # Convert to numpy arrays and calculate metrics
            all_pred = np.array(all_pred)
            all_gt = np.array(all_gt)
            all_probs = np.array(all_probs)
            
            # Print metrics
            print(classification_report(all_gt, all_pred))
            tn, fp, fn, tp = confusion_matrix(all_gt, all_pred).ravel()
            score = roc_auc_score(all_gt, all_probs)
            print(f"ROC AUC score: {score}")
            if score > record:
                print("New record, saving model...")
                torch.save(net.state_dict(), best_model_path)
                record = score
    net.load_state_dict(torch.load(best_model_path))
    return net, record
    # first step: weighted CE loss; consider doing normalization
    # second step: gaussian target weighting, label propagation (propagate labels through bright patches to get better annotations)
    # and incorporate classification loss in the loss function (CE loss of prob of greatest probability pixel wrt whether asteroid is present or not)
    # compare performance with deepstreaks (2019) and that 2025 paper
    # https://arxiv.org/html/2405.14238v1 
    # focal loss on the more challenging negatives (assign greater weight to high confidence positive, yet negative pixels)
    
    # an idea from deepstreaks or other papers: use curriculum learning
    # start with easier samples with longer streaks (larger streak area)
    # and then go to harder ones with shorter streaks and negative samples