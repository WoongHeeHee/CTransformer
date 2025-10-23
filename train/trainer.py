import torch
import torch.nn as nn
import torch.optim as optim
from model.complex_linear import ComplexLinear


class Trainer:
    """
    Trainer class for CTransformer model
    """
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_epoch(self, dataloader, criterion, optimizer):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            src_mask = None  # Assuming no mask for simplicity
            output = self.model(batch_x, src_mask)
            output = output.mean(dim=1)
            output = ComplexLinear(self.model.layers[0].self_attention.d_model, 1)(output)
            output = abs(output)
            output = torch.sigmoid(output)  # Apply sigmoid for binary classification
            output = output.squeeze(-1)

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def train(self, dataloader, num_epochs=100, learning_rate=0.01):
        """
        Complete training loop
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, criterion, optimizer)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
        print("Training complete.")
        
    def evaluate(self, dataloader):
        """
        Evaluate the model
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                src_mask = None
                output = self.model(batch_x, src_mask)
                output = output.mean(dim=1)
                output = ComplexLinear(self.model.layers[0].self_attention.d_model, 1)(output)
                output = abs(output)
                output = torch.sigmoid(output).squeeze(-1)
                
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (output > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
