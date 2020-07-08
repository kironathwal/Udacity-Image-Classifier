# PROGRAMMER: Kiron Athwal
# DATE CREATED: 03/07/20
# REVISED DATE: 07/07/20
# PURPOSE: Train a new network on a data set

### import libraries ###

import argparse #03/07 needed for function 1
import torch
from torch import nn, optim #03/07 needed for function 5,6,7
from torchvision import datasets, transforms, models #03/07 needed for function 2,3,4 
from collections import OrderedDict #03/07 needed for function 5


### define different functions ###

## function number 1: uses argparse for command-line options

def get_input_args(): #03/07
    
    parser = argparse.ArgumentParser(description = 'Train a new network on a dataset')
    # argument 1: path to directory with image datasets
    parser.add_argument('data_dir', type = str, default='flowers', help = 'Path to the folder of image dataset')
    # argument 2: set directory to save checkpoints
    parser.add_argument('--save_dir', type = str, default='./checkpoint.pth', help = 'Set directory to save checkpoints')
    # argument 3: choose architecture ie. which CNN model
    parser.add_argument('--arch', type = str, default='vgg16', help = 'Choose which CNN model to use - vgg16 or densenet161')
    # argument 4: set hyperparameter learning_rate
    parser.add_argument('--learning_rate', type = float, default=0.001, help = 'Choose learning rate')
    # argument 5: set hyperparameter hidden_units
    parser.add_argument('--hidden_units', type = int, default=1024, help = 'Choose no of hidden units in 1st layer')
    # argument 6: set hyperparameter epochs
    parser.add_argument('--epochs', type = int, default=5, help = 'Choose no of epochs')
    # argument 7: set dropout probability
    parser.add_argument('--dropout', type = int, default=0.3, help = 'Choose dropout probability')
    # argument 8: use GPU for training
    parser.add_argument('--gpu', type = str, default='gpu', help = 'Use GPU for training')
    
    args = parser.parse_args()
    
    return args
    
## function number 2: transforms for the training sets (including rotations and horizontal flips)

def train_transformation(train_dir): #03/07
    # data transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])   
    # load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
                                          
    # define the dataloader
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    return trainloader, train_data                                          
                                          
## function number 3: transforms for the testing/validation sets (no rotations or scaling)

def test_transformation(test_dir): #03/07
    # data transforms                                      
    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    # load datasets with ImageFolder
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
                                         
    # define the dataloader
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    return testloader 
                                         
## function number 4: load a pre-trained CNN e.g. vgg16

def load_pretrained_model(arch): #03/07

    model = getattr(models, arch)(pretrained=True) #07/07
    #model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    in_feature = model.classifier[0].in_features #07/07
    
    return model, in_feature
                                         
## function number 5: build the classifier structure

def classifier_structure(model, dropout, hidden_units, learning_rate, device, in_feature): #03/07
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_feature, hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(dropout)),
                                            ('fc2', nn.Linear(hidden_units, 256)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout(dropout)),
                                            ('fc3', nn.Linear(256,102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    model.to(device);

    return model, criterion, optimizer
                               
## function number 6: train the model,and track loss and accuracy on validation set

def train_model(trainloader, model, device, epochs, criterion, optimizer, validloader): #03/07

    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Training loss: {running_loss/print_every:.3f}.."
                      f"Validity loss: {valid_loss/len(validloader):.3f}.."
                      f"Validity accuracy: {accuracy*100/len(validloader):.3f} %")
                running_loss = 0
                model.train()
    print("Training process completed")
    
    return model        
                               
## function 7: test the model on test set

def test_model(testloader, model, device, criterion): #03/07
                               
    accuracy = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            test_loss += criterion(log_ps, labels).item()
            ps = torch.exp(log_ps)  
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Testing loss: {test_loss/len(testloader):.3f}.."
          f"Testing accuracy: {accuracy*100/len(testloader):.3f} %")

    return
                             
## function 8: save the checkpoint

def save_checkpoint(model, train_data, epochs, dropouts, optimizer, save_dir, learning_rate): #03/07
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'epochs': epochs,
                  'dropouts': dropouts,
                  'learning_rate': learning_rate,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    print("Model saved successfully")

    return
                                                                                         
                                         
### define the main functions ###

def main():
    
    ## function 1: command line options
    args = get_input_args()
    
    ## setting the directories for training 
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 
    
    ## function 2: transforms for training set
    trainloader, train_data = train_transformation(train_dir)
    
    ## function 3: transforms for validation and training set
    validloader = test_transformation(valid_dir)
    testloader = test_transformation(test_dir)
                                         
    ## Use the GPU if available, otherwise use CPU
    if args.gpu :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                         
    ## function 4: load a pre-trained CNN
    model, in_feature = load_pretrained_model(args.arch)

    ## function 5: build the classifier structure
    model, criterion, optimizer = classifier_structure(model, args.dropout, args.hidden_units, args.learning_rate, device, in_feature)   

    ## function 6: train the model, and track loss and accuracy on validation set                                  
    model = train_model(trainloader, model, device, args.epochs, criterion, optimizer, validloader)

    ## function 7: test the model on test set
    test_model(testloader, model, device, criterion)

    ## function 8: save the checkpoint

    save_checkpoint(model, train_data, args.epochs, args.dropout, optimizer, args.save_dir, args.learning_rate)

### call to main function to run the program ###

if __name__ == "__main__":
    main()
    