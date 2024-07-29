import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
from new_model import NewModel
from dataset import get_dataloader
import os
from torch.optim.lr_scheduler import StepLR

# Assuming your get_dataloader function returns a DataLoader correctly
train_loader = get_dataloader(root_dir="final_daataa", batch_size=15)
epochs = 500
epochs_done = 0
for files in os.listdir("weights"):
    name,extension = os.path.splitext(files)
    epoch_number = int(name[12:])
    if epoch_number > epochs_done:
        epochs_done = epoch_number

model = NewModel()
model.load_state_dict(torch.load(f'weights/model_epoch_{epochs_done}.pth'))
print(f"Loading weights from weights/model_epoch_{epochs_done}.pth")
model.model_tokenizer.requires_grad_(False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9,weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    if epoch < epochs_done:
        continue
    
    running_loss = 0.0
    tqdm_train_loader = tqdm(train_loader, total=len(train_loader))
    
    for i, data in enumerate(tqdm_train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        tqdm_train_loader.set_description(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}, Running Loss: {running_loss / (i + 1):.4f}')

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        torch.save(model.state_dict(), f'weights/model_epoch_{epoch + 1}.pth')        

print('Finished Training')

