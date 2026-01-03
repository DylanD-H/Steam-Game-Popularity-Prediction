import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle

class SteamPreprocessor:
    def __init__(self, alpha=5):
        self.alpha = alpha
        self.overall_mean = None
        self.dev_target = None
        self.pub_target = None
        self.genre_columns = None
        self.category_columns = None
        self.n_features = None
        self.dv = None

    @staticmethod
    def parse_release_date(date_str):
        date_str = str(date_str).strip()
        if date_str.upper().startswith("Q"):
            quarter, year = date_str.upper().split()
            year = int(year)
            if quarter == "Q1":
                return pd.Timestamp(year=year, month=2, day=15)
            elif quarter == "Q2":
                return pd.Timestamp(year=year, month=5, day=15)
            elif quarter == "Q3":
                return pd.Timestamp(year=year, month=8, day=15)
            elif quarter == "Q4":
                return pd.Timestamp(year=year, month=11, day=15)
        elif date_str.isdigit() and len(date_str) == 4:
            year = int(date_str)
            return pd.Timestamp(year=year, month=6, day=30)
        return pd.to_datetime(date_str)

    @staticmethod
    def encode_multi(items, mapping, overall_mean):
        items = str(items).split(';')
        encodings = [mapping.get(i, overall_mean) for i in items]
        return sum(encodings) / len(encodings)

    def fit(self, df):
        df = df.copy()

        df.genres = df.genres.fillna('N/A')
        df.categories = df.categories.fillna('N/A')
        df.developer = df.developer.fillna('N/A')
        df.publisher = df.publisher.fillna('N/A')

        df["successful"] = (df["recommendations"] > 0).astype(int)

        df['release_date_parsed'] = df['release_date'].apply(self.parse_release_date)
        df['release_month'] = df['release_date_parsed'].dt.month
        df['release_day_of_week'] = df['release_date_parsed'].dt.dayofweek

        genres_expanded = df['genres'].str.get_dummies(sep=';').add_prefix('genre_')
        categories_expanded = df['categories'].str.get_dummies(sep=';').add_prefix('cat_')
        df = pd.concat([df, genres_expanded, categories_expanded], axis=1)

        self.genre_columns = genres_expanded.columns.tolist()
        self.category_columns = categories_expanded.columns.tolist()

        df['num_genres'] = df['genres'].str.count(';') + 1
        df['num_categories'] = df['categories'].str.count(';') + 1

        self.overall_mean = df['successful'].mean()

        self.dev_target = df.groupby('developer').agg(
            successes=('successful', 'sum'),
            total=('successful', 'count')
        )
        self.dev_target['developer_encoded'] = (
            (self.dev_target['successes'] + self.alpha * self.overall_mean) /
            (self.dev_target['total'] + self.alpha)
        )

        self.pub_target = df.groupby('publisher').agg(
            successes=('successful', 'sum'),
            total=('successful', 'count')
        )
        self.pub_target['publisher_encoded'] = (
            (self.pub_target['successes'] + self.alpha * self.overall_mean) /
            (self.pub_target['total'] + self.alpha)
        )

        df['developer_encoded'] = df['developer'].apply(
            lambda x: self.encode_multi(x, self.dev_target['developer_encoded'], self.overall_mean)
        )
        df['publisher_encoded'] = df['publisher'].apply(
            lambda x: self.encode_multi(x, self.pub_target['publisher_encoded'], self.overall_mean)
        )

        df = df.drop(columns=['genres', 'categories', 'release_date', 'release_date_parsed', 'name', 'developer', 'publisher', 'recommendations'])

        df_dicts = df.drop(columns=['successful']).to_dict(orient='records')

        self.dv = DictVectorizer(sparse=False)
        X = self.dv.fit_transform(df_dicts)
        self.n_features = X.shape[1]
        y = df['successful'].values

        return X, y

    def transform(self, df):
        df = df.copy()
        
        df.genres = df.genres.fillna('N/A')
        df.categories = df.categories.fillna('N/A')
        df.developer = df.developer.fillna('N/A')
        df.publisher = df.publisher.fillna('N/A')

        df["successful"] = (df.get("recommendations", 0) > 0).astype(int)

        df['release_date_parsed'] = df['release_date'].apply(self.parse_release_date)
        df['release_month'] = df['release_date_parsed'].dt.month
        df['release_day_of_week'] = df['release_date_parsed'].dt.dayofweek

        genres_expanded = df['genres'].str.get_dummies(sep=';').add_prefix('genre_')
        categories_expanded = df['categories'].str.get_dummies(sep=';').add_prefix('cat_')

        genres_expanded = genres_expanded.reindex(columns=self.genre_columns, fill_value=0)
        categories_expanded = categories_expanded.reindex(columns=self.category_columns, fill_value=0)

        df = pd.concat([df, genres_expanded, categories_expanded], axis=1)

        df['num_genres'] = df['genres'].str.count(';') + 1
        df['num_categories'] = df['categories'].str.count(';') + 1

        df['developer_encoded'] = df['developer'].apply(
            lambda x: self.encode_multi(x, self.dev_target['developer_encoded'], self.overall_mean)
        )
        df['publisher_encoded'] = df['publisher'].apply(
            lambda x: self.encode_multi(x, self.pub_target['publisher_encoded'], self.overall_mean)
        )

        df = df.drop(columns=['genres', 'categories', 'release_date', 'release_date_parsed', 'name','developer', 'publisher','recommendations'], errors='ignore')

        df_dicts = df.drop(columns=['successful'], errors='ignore').to_dict(orient='records')
        X = self.dv.transform(df_dicts)

        return X, df['successful'].values

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        return self
