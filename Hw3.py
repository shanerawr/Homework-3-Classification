import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# load and prepare
df = pd.read_csv("course_lead_scoring_2026.csv")
categorical = ["lead_source", "industry", "employment_status", "location"]
numerical = ["annual_income", "number_of_courses_viewed", "interaction_count", "lead_score"]
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)

#q1: mode of industry
print('Q1: ', df['industry'].mode()[0])

#Q2: correlation matrix
corr = df[numerical].corr()
print(corr)

pairs = [
    ('interaction_count','lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

for a,b in pairs:
    print('Q2: ', a, b, round(corr.loc[a,b], 4))

#split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=42)

y_train = df_train['converted'].values
y_val = df_val['converted'].values

#remove the target from the feature dfs
del df_train['converted']
del df_val['converted']
del df_test['converted']

#q3: mutual information
for c in categorical:
    score = mutual_info_score(df_train[c],y_train)
    print('Q3: ', c, round(score, 2))

#helper: one-hot encode, train, return validation accuracy
def fit_eval(features, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))
    X_val = dv.transform(df_val[features].to_dict(orient='records'))
    model = LogisticRegression(solver='liblinear', C=C,max_iter=1000, random_state=42)
    model.fit(X_train,y_train)
    return accuracy_score(y_val, model.predict(X_val))

all_features = categorical + numerical

#q4: baseline logistics regression
base_acc = fit_eval(all_features)
print('Q4:', round(base_acc, 2))

#q5: feature elimination
for f in all_features:
    acc = fit_eval([x for x in all_features if x != f])
    print('Q5: ', f, 'diff = ', base_acc-acc)

#q6: regularization
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    print('Q6:', C, round(fit_eval(all_features, C=C), 3))
