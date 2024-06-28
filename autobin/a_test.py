import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_binary_classifier(train_x1, train_age, train_stage_i, train_stage_ii, train_race_white, train_yevent,
                            eval_x1, eval_age, eval_stage_i, eval_stage_ii, eval_race_white, eval_yevent,
                            input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2, Learning_Rate, L2, epoch_num, patience,
                            path="saved_model/binary_classifier_checkpoint.pt"):
    net = BinaryClassifier(input_n, level_2_dim, Dropout_Rate_1, Dropout_Rate_2)

    early_stopping = EarlyStopping(patience=patience, verbose=False, path=path)

    if torch.cuda.is_available():
        net.cuda()
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2)

    # Calculate the class weights
    num_pos = train_yevent.sum()
    num_neg = train_yevent.shape[0] - num_pos
    pos_weight = num_neg / (num_pos + num_neg)
    neg_weight = num_pos / (num_pos + num_neg)

    start_time = time.time()
    train_losses = []
    val_accuracies = []
    val_aucs = []
    val_f1s = []
    
    for epoch in range(epoch_num):
        net.train()
        opt.zero_grad()

        y_pred = net(train_x1, train_age, train_stage_i, train_stage_ii, train_race_white, s_dropout=True)

        # Calculate balanced cross-entropy loss
        loss_pos = -pos_weight * (train_yevent * torch.log(torch.sigmoid(y_pred)).squeeze())
        loss_neg = -neg_weight * ((1 - train_yevent) * torch.log(1 - torch.sigmoid(y_pred)).squeeze())
        loss = (loss_pos + loss_neg).mean()

        loss.backward()
        opt.step()

        train_losses.append(loss.item())

        net.eval()
        eval_y_pred = net(eval_x1, eval_age, eval_stage_i, eval_stage_ii, eval_race_white, s_dropout=False)
        eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy().round())
        eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy())
        eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy().round())
        
        val_accuracies.append(eval_accuracy)
        val_aucs.append(eval_auc)
        val_f1s.append(eval_f1)

        early_stopping(eval_auc, net)  # You can also use eval_f1 here
        if early_stopping.early_stop:
            print("Early stopping, number of epochs: ", epoch)
            print('Save model of Epoch {:d}'.format(early_stopping.best_epoch_num))
            break
        if (epoch+1) % 100 == 0:
            net.eval()
            train_y_pred = net(train_x1, train_age, train_stage_i, train_stage_ii, train_race_white, s_dropout=False)
            train_accuracy = accuracy_score(train_yevent.detach().cpu().numpy(), torch.sigmoid(train_y_pred).detach().cpu().numpy().round())
            print("Training Accuracy: %s," % train_accuracy, "validation Accuracy: %s." % eval_accuracy)

    print("Loading model, best epoch: %s." % early_stopping.best_epoch_num)
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    net.eval()
    train_y_pred = net(train_x1, train_age, train_stage_i, train_stage_ii, train_race_white, s_dropout=False)
    train_accuracy = accuracy_score(train_yevent.detach().cpu().numpy(), torch.sigmoid(train_y_pred).detach().cpu().numpy().round())

    net.eval()
    eval_y_pred = net(eval_x1, eval_age, eval_stage_i, eval_stage_ii, eval_race_white, s_dropout=False)
    eval_accuracy = accuracy_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy().round())
    eval_auc = roc_auc_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy())
    eval_f1 = f1_score(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy().round())

    print("Final training Accuracy: %s," % train_accuracy, "final validation Accuracy: %s." % eval_accuracy)
    print("Final validation AUC: %s," % eval_auc, "final validation F1: %s." % eval_f1)
    time_elapse = np.array(time.time() - start_time).round(2)
    print("Total time elapse: %s." % time_elapse)

    # Plotting the metrics
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs, val_aucs, 'g', label='Validation AUC')
    plt.plot(epochs, val_f1s, 'm', label='Validation F1')
    plt.title('Validation AUC and F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    # Confusion Matrix
    plt.subplot(1, 4, 4)
    conf_matrix = confusion_matrix(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy().round())
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(eval_yevent.detach().cpu().numpy(), torch.sigmoid(eval_y_pred).detach().cpu().numpy())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return (train_y_pred, eval_y_pred, train_accuracy, eval_accuracy, eval_auc)
