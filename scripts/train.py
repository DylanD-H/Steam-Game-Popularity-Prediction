import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from preprocess import SteamPreprocessor

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


# def parse_release_date(date_str):

#     date_str = str(date_str).strip()
    
#     if date_str.upper().startswith("Q"):
#         quarter, year = date_str.upper().split()
#         year = int(year)
#         if quarter == "Q1":
#             return pd.Timestamp(year=year, month=2, day=15)
#         elif quarter == "Q2":
#             return pd.Timestamp(year=year, month=5, day=15)
#         elif quarter == "Q3":
#             return pd.Timestamp(year=year, month=8, day=15)
#         elif quarter == "Q4":
#             return pd.Timestamp(year=year, month=11, day=15)
    
#     elif date_str.isdigit() and len(date_str) == 4:
#         year = int(date_str)
#         return pd.Timestamp(year=year, month=6, day=30)
    

#     return pd.to_datetime(date_str)

# def encode_multi(items, mapping, overall_mean):
#     items = str(items).split(';')
#     encodings = [mapping.get(i, overall_mean) for i in items]
#     return sum(encodings) / len(encodings)


# df.genres = df.genres.fillna('N/A')
# df.categories = df.categories.fillna('N/A')
# df.developer = df.developer.fillna('N/A')
# df.publisher = df.publisher.fillna('N/A')

# df["successful"] = (df["recommendations"] > 0).astype(int)

# df['release_date_parsed'] = df['release_date'].apply(parse_release_date)
# df['release_month'] = df['release_date_parsed'].dt.month
# df['release_day_of_week'] = df['release_date_parsed'].dt.dayofweek


# genres_expanded = df['genres'].str.get_dummies(sep=';').add_prefix('genre_')
# categories_expanded = df['categories'].str.get_dummies(sep=';').add_prefix('cat_')
# df = pd.concat([df, genres_expanded, categories_expanded], axis=1)

# df['num_genres'] = df['genres'].str.count(';') + 1
# df['num_categories'] = df['categories'].str.count(';') + 1
# df = df.drop(columns=['genres', 'categories', 'release_date', 'release_date_parsed', 'name'])




# alpha = 5 
# overall_mean = df_full_train['successful'].mean()

# dev_target = df_full_train.groupby('developer').agg(
#     successes=('successful', 'sum'),
#     total=('successful', 'count')
# )
# dev_target['developer_encoded'] = (dev_target['successes'] + alpha * overall_mean) / (dev_target['total'] + alpha)

# pub_target = df_full_train.groupby('publisher').agg(
#     successes=('successful', 'sum'),
#     total=('successful', 'count')
# )
# pub_target['publisher_encoded'] = (pub_target['successes'] + alpha * overall_mean) / (pub_target['total'] + alpha)


# df_full_train['developer_encoded'] = df_full_train['developer'].apply(
#     lambda x: encode_multi(x, dev_target['developer_encoded'], overall_mean)
# )
# df_full_train['publisher_encoded'] = df_full_train['publisher'].apply(
#     lambda x: encode_multi(x, pub_target['publisher_encoded'], overall_mean)
# )

# y_train = df_full_train.successful.values
# train_recs = df_full_train.recommendations.values

# df_full_train = df_full_train.drop(columns=['developer', 'publisher', 'successful', 'recommendations'])


# dv = DictVectorizer()
# train_dict = df_full_train.to_dict(orient='records')
# X_train = dv.fit_transform(train_dict)

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
    