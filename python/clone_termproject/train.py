import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import BasicBLock, Bottleneck, ResNet, ResNet18

def acc(pred, label) :
    pred = pred.argmax(dim-1)
    retrun torch.sum(pred == label).item()

def train(args, data_loader, model) :
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs) :
        train_losses = []
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch+1} / {args.epochs}]")

        prev_accuracy = 0

        model.train()

        pbar = tqdm(data_loader)
        #x : images, y : labels      
        for i, (x, y) in enumerate(pbar) :
            image = x.to(args.device)
            label = y.to(args.device)
            
            optimizer.zero_grad()

            output = model(image)

            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)
        
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total

        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))

        torch.save(model.state_dict(), f'{args.save_path}/model.pth')