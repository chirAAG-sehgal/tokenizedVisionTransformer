import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm
from new_model import NewModel
from dataset import get_dataloader

# Assuming your get_dataloader function returns a DataLoader correctly
train_loader = get_dataloader(root_dir="final_daataa", batch_size=32)

model = NewModel()
model.model_tokenizer.requires_grad_(False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 1500

for epoch in range(epochs):
    running_loss = 0.0
    tqdm_train_loader = tqdm(train_loader, total=len(train_loader))
    
    for i, data in enumerate(tqdm_train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        tqdm_train_loader.set_description(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / (i + 1):.4f}')

    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    if (epoch + 1) % 5 == 0:  # Save every 50 epochs
        torch.save(model.state_dict(), f'weights/model_epoch_{epoch + 1}.pth')

print('Finished Training')

