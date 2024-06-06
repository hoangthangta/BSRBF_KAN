import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import EfficientKAN, FastKAN, BSRBF_KAN, Faster_KAN
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from file_io import *

def run(model_name = 'bsrbf_kan', batch_size = 64, n_input = 28*28, epochs = 10, n_output = 10, n_hidden = 64, \
        grid_size = 5, num_grids = 8, spline_order = 3):

    start = time.time()
    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Create model storage
    output_path = 'output/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    saved_model_name = model_name + '_mnist.pth'
    saved_model_history =  model_name + '_mnist.json'
    with open(os.path.join(output_path, saved_model_history), 'w') as fp: pass

    # Define model
    model = {}
    print('model_name: ', model_name)
    if (model_name == 'bsrbf_kan'):
        model = BSRBF_KAN([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order)
    elif(model_name == 'fast_kan'):
        model = FastKAN([n_input, n_hidden, n_output], num_grids = num_grids)
    elif(model_name == 'faster_kan'):
        model = FasterKAN([n_input, n_hidden, n_output], num_grids = num_grids)
    else:
        model = EfficientKAN([n_input, n_hidden, n_output], grid_size = grid_size, spline_order = spline_order)
    model.to(device)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()

    best_accuracy, best_epoch = 0, 0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_accuracy, train_loss = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, n_input).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
        
        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
            
        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, n_input).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Update learning rate
        scheduler.step()
        
        # Choose best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model, output_path + '/' + saved_model_name)
                    
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        print(f"Epoch {epoch}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'best_accuracy': best_accuracy, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
    
    end = time.time()
    print(f"Training time (s): {end-start}")
    write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')
    
def main(args):
    if (args.mode == 'train'):
        run(model_name = args.model_name, batch_size = args.batch_size, epochs = args.epochs, \
            n_input = args.n_input, n_output = args.n_output, n_hidden = args.n_hidden, \
            grid_size = args.grid_size, num_grids = args.num_grids, spline_order = args.spline_order)
                    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_input', type=int, default=28*28)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='output/model.pth')
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--spline_order', type=int, default=3)
    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)
    
#python run_mnist.py --mode "train" --model_name "bsrbf_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3

#python run_mnist.py --mode "train" --model_name "efficient_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3

#python run_mnist.py --mode "train" --model_name "fast_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8

#python run_mnist.py --mode "train" --model_name "faster_kan" --epochs 15 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8
