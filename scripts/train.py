import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .preprocess import SteamPreprocessor

class SteamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SteamNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


if __name__ == "__main__":

    df = pd.read_csv('./steam_data.csv', index_col=0)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    with open("./model/steam_test_df.pkl", "wb") as f:
        pickle.dump(df_test, f)
        
    preprocessor = SteamPreprocessor()
    X_train, y_train = preprocessor.fit(df_full_train)  
    
    
    train_ds = SteamDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SteamNN(X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


    torch.save(model.state_dict(), "./model/steam_nn.pth")
    preprocessor.save("./model/steam_preprocessor.pkl")
    