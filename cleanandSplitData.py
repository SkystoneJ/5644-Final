import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import re
import string
from sklearn.feature_extraction import text
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def remove_specific_words(text, words_to_remove):
    pattern = r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b'
    text = re.sub(pattern, '', text)
    return text


def clean_text(text, words_to_remove):
    text = text.lower()
    text = remove_specific_words(text, words_to_remove)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = text.strip()
    return text


data_df = pd.read_csv('listings.csv')
threshold = 0.5 * len(data_df)
data_df = data_df.dropna(thresh=threshold, axis=1)
stop_words = text.ENGLISH_STOP_WORDS
data_df['price'] = data_df['price'].replace('[\$,]', '', regex=True).astype(float)
sns.histplot(data_df['price'], bins=100, kde=True)
plt.title('Price Distribution')
plt.show()
price_skewness = data_df['price'].skew()
print("skewness:", price_skewness)
columns_to_drop = ['listing_url', 'id', 'host_name', 'scrape_id', 'last_scraped', 'picture_url', 'host_id',
                   'host_url', 'host_name', 'host_since', 'host_thumbnail_url', 'host_picture_url',
                   'host_listings_count', 'host_total_listings_count', 'host_verifications', 'latitude', 'longitude',
                   'has_availability', 'first_review', 'last_review', 'host_about']
data_df = data_df.drop(columns=columns_to_drop)
for column in data_df.select_dtypes(include=['int64', 'float64']).columns:
    data_df[column].fillna(data_df[column].mean(), inplace=True)

# fill up the category
for column in data_df.select_dtypes(include=['object']).columns:
    mode_value = data_df[column].mode()[0]
    data_df[column].fillna(mode_value, inplace=True)
data_df['host_response_rate'] = data_df['host_response_rate'].str.rstrip('%').astype('float') / 100.0
data_df['host_acceptance_rate'] = data_df['host_acceptance_rate'].str.rstrip('%').astype('float') / 100.0
data_df['host_is_superhost'] = data_df['host_is_superhost'].map({'t': 1, 'f': 0})
data_df['host_has_profile_pic'] = data_df['host_has_profile_pic'].map({'t': 1, 'f': 0})
data_df['host_identity_verified'] = data_df['host_identity_verified'].map({'t': 1, 'f': 0})
data_df['instant_bookable'] = data_df['instant_bookable'].map({'t': 1, 'f': 0})
words_to_remove = ['specific', 'unnecessary', 'redundant', 'very', '<br/>']
data_df['description'] = data_df['description'].apply(lambda x: clean_text(x, words_to_remove))
data_df['neighborhood_overview'] = data_df['neighborhood_overview'].apply(lambda x: clean_text(x, words_to_remove))
data_df['amenities'] = data_df['amenities'].apply(lambda x: clean_text(x, words_to_remove))
data_df['room_type'] = data_df['room_type'].apply(lambda x: clean_text(x, words_to_remove))
data_df['neighbourhood'] = data_df['neighbourhood'].apply(lambda x: clean_text(x, words_to_remove))
data_df['host_location'] = data_df['host_location'].apply(lambda x: clean_text(x, words_to_remove))
data_df['name'] = data_df['name'].apply(lambda x: clean_text(x, words_to_remove))
print(data_df.dtypes)
categorical_columns = ['source', 'host_response_time', 'host_location',
                       'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed',
                       'property_type', 'room_type', 'bathrooms_text']
final_drop_columns = ['source', 'host_response_time', 'host_location',
                       'host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed',
                       'property_type', 'room_type', 'bathrooms_text', 'combined_text', 'description',
                      'neighborhood_overview', 'name', 'calendar_last_scraped', 'amenities']

data_df['combined_text'] = data_df['description'] + ' ' + data_df['neighborhood_overview'] + ' ' + data_df['name']  + \
                           ' ' + data_df['amenities']

# 3. vectorized by using One Hot Encoder
ohe = OneHotEncoder()
category_encoded = ohe.fit_transform(data_df[categorical_columns])

# 4. vectorized by using TF-IDF
tfidf = TfidfVectorizer()
text_tfidf = tfidf.fit_transform(data_df['combined_text'])

print('Shape of category_encoded:', category_encoded.shape)
print('Shape of text_tfidf:', text_tfidf.shape)
features = sparse.hstack((category_encoded, text_tfidf))
print("Combined feature matrix shape:", features.shape)


category_feature_names = ohe.get_feature_names_out(categorical_columns)
text_feature_names = tfidf.get_feature_names_out()


# features_df = pd.DataFrame(category_encoded.toarray(), columns=category_feature_names)
# combined_df = pd.concat([data_df.reset_index(drop=True), features_df], axis=1)
# output_df = combined_df.drop(columns = final_drop_columns)
# correlations = output_df.corrwith(output_df['price']).sort_values(ascending=False)
# high_corr_features = correlations[abs(correlations) > 0.1]
# filtered_data = output_df[high_corr_features.index.tolist()]
# corr_matrix = filtered_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
# plt.title('Correlation Matrix Heatmap')
# plt.show()
# combined_df.to_csv('data.csv', index=False)
# output_df.to_csv('output.csv', index=False)

all_feature_names = list(category_feature_names) + list(text_feature_names)
features_df = pd.DataFrame(features.toarray(), columns=all_feature_names)
combined_df = pd.concat([data_df.reset_index(drop=True), features_df], axis=1)
output_df = combined_df.drop(columns = final_drop_columns)

X = output_df.drop('price', axis=1)
y = output_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)
X_train.to_csv('trainX.csv', index=False)
y_train.to_csv('trainy.csv', index=False)
X_test.to_csv('testX.csv', index=False)
y_test.to_csv('testy.csv', index=False)