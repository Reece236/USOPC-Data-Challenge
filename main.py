import pandas as pd
import numpy as np
import pylab as py
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define medal means for gymnastics apparatus based on past Olympic performances
medal_means = {'w': {'BB': 15.27, 'FX': 15.19, 'UB': 15.85, 'VT': 15.38},
               'm': {'PH': 15.73, 'SR': 15.87, 'VT': 15.61, 'PB': 15.99, 'HB': 15.76, 'FX': 15.54}}


# Adjust scores for each gymnast based on historical medal means
def adjust_score(row, y):
    try:
        mean = np.mean(list(medal_means[row['Gender']].values()))
        adj = mean / medal_means[row['Gender']][row['Apparatus']]
        return row[y] * adj
    except KeyError:
        return np.nan

# Plot feature importance
def plot_feature_importance(model, label, X):
    feature_importance = pd.DataFrame({'feature': list(X.columns), 'importance': model.feature_importances_}).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance for {label}')
    plt.tight_layout()
    plt.show()


# Create a linear regression model and print scores
def linear_regression(data, y, label='-'):
    # Drop rows with NaN values in the target variable
    data = data.dropna(subset=[f'last_{i}_{y}' for i in range(1, 7)] + ['Score'])

    # Check if there are still enough data points after dropping NaN values
    if len(data) < 2:
        print(f'Not enough data for linear regression using {label} Data. Skipping...')
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        data[[f'last_{i}_{y}' for i in range(1, 7)]], data[y],
        test_size=0.33, random_state=6)

    lm = LinearRegression()

    try:
        lm.fit(X_train, y_train)
    except ValueError as e:
        print(f'Error fitting linear regression model using {label} Data. Skipping...')
        print(e)
        return None

    y_pred = lm.predict(X_test)

    # Check assumptions

    # 1) Independence
    errors = y_test - y_pred
    plt.scatter(range(len(y_test)), errors)
    plt.title('Independence Check')
    plt.xlabel('index')
    plt.ylabel('errors')
    plt.show()

    # 2) Linearity
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.title('Linearity Check')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.show()

    # 3) Normality
    stats.probplot(errors, dist="norm", plot=py)
    py.title('Normality Check')
    py.show()

    print(f'\nResults for {y} using {label} Data:')
    print('Coefficients: \n', lm.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

    # Get Score for each Apparatus
    df = data.copy()
    df['pred'] = lm.predict(df[[f'last_{i}_{y}' for i in range(1, 7)]])

    for apparatus in data['Apparatus'].unique():
        y_pred = df.loc[df['Apparatus'] == apparatus]['pred']
        y_true = df.loc[df['Apparatus'] == apparatus]['Score']
        mse = mean_squared_error(y_true, y_pred)
        print(label, apparatus, mse)

    # Plot feature importance
    plot_feature_importance(lm, label, X_train)

    return lm


# Find the best pairing of gymnasts for a given apparatus and cutoff score
def find_best_pairing(data, y, cut_off=12, n=5):
    start = time.time()
    gymnasts = data.loc[data[y] > cut_off]['id'].unique()
    all_combinations = list(combinations(gymnasts, n))

    top_combinations = []
    top_scores = []

    for i, combination in enumerate(all_combinations):
        current_combination_df = data[data['id'].isin(combination)]
        total_score = current_combination_df.groupby('Apparatus')[y].nlargest(3).sum().sum()

        if len(top_combinations) < 3 or total_score > min(top_scores):
            top_combinations.append(combination)
            top_scores.append(total_score)

            top_combinations = [c for _, c in sorted(zip(top_scores, top_combinations), reverse=True)[:3]]
            top_scores = sorted(top_scores, reverse=True)[:3]

        if i == 10:
            now = time.time()
            eta = (now-start)/ 10 * len(all_combinations)
            print(f'{eta/60} minutes remaining')


    return top_combinations, top_scores


if __name__ == "__main__":
    # Read and combine data from two CSV files
    df = pd.concat([pd.read_csv('data_2022_2023.csv'),
                    pd.read_csv('data_2017_2021.csv')])
    df.reset_index(drop=True, inplace=True)


    # Add gymnast identifier and reformat date
    df['id'] = df['LastName'] + df['FirstName'] + df['Gender'] + df['Country']
    df['Date'] = df['Date'].apply(lambda x: x.split('-')[1] if '-' in x else x)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Rename apparatus names
    df['Apparatus'] = df['Apparatus'].apply(lambda x: 'VT' if 'VT' in x else x)
    df['Apparatus'] = df['Apparatus'].apply(lambda x: 'HB' if x == 'hb' else x)
    df['Apparatus'] = df['Apparatus'].apply(lambda x: 'UB' if x == 'UE' else x)

    for i in range(1, 7):
        df[f'last_{i}_Score'] = df.groupby(['id', 'Apparatus'])['Score'].shift(i)

    # Drop rows with null values
    lm_data = df.fillna(0)


    # Create separate datasets for men and women and split by apparatus
    men_df = lm_data.loc[lm_data['Gender'] == 'm']
    women_df = lm_data.loc[lm_data['Gender'] == 'w']

    # Perform linear regression for both men and women
    men_lm = linear_regression(men_df, 'Score', 'Men')
    women_lm = linear_regression(women_df, 'Score', 'Women')

    # Predict scores for US gymnasts
    usa_w = df.loc[(df['Country'] == 'USA') & (df['Gender'] == 'w')].groupby(['id', 'Apparatus']).last().reset_index()
    usa_m = df.loc[(df['Country'] == 'USA') & (df['Gender'] == 'm')].groupby(['id', 'Apparatus']).last().reset_index()

    usa_w['pred_Score'] = women_lm.predict(
        np.array(usa_w[['Score'] + [f'last_{i}_Score' for i in range(1, 6)]].fillna(0)))
    usa_m['pred_Score'] = men_lm.predict(
        np.array(usa_m[['Score'] + [f'last_{i}_Score' for i in range(1, 6)]].fillna(0)))

    # Calculate adjusted scores
    usa_w['pred_adjScore'] = usa_w.apply(lambda row: adjust_score(row, y='pred_Score'), axis=1)
    usa_m['pred_adjScore'] = usa_m.apply(lambda row: adjust_score(row, y='pred_Score'), axis=1)

    # Find the best gymnast pairings for men and women
    men_adj_teams, men_adj_scores = find_best_pairing(usa_m, 'pred_adjScore', cut_off=13.7)
    men_teams, men_scores = find_best_pairing(usa_m, 'pred_Score', cut_off=13.7)
    women_adj_teams, women_adj_scores = find_best_pairing(usa_w, 'pred_adjScore', cut_off=13.7)
    women_teams, women_scores = find_best_pairing(usa_w, 'pred_Score', cut_off=13.7)

    # Print the top teams for men and women
    print("\nMen's Teams")
    for team, score in zip(men_teams, men_scores):
        print(team, score)
    print("\n Men's Adjusted Teams")
    for team, score in zip(men_adj_teams, men_adj_scores):
        print(team, score)
    print("\n Women's Teams")
    for team, score in zip(women_teams, women_scores):
        print(team, score)
    print("\n Women's Adjusted Teams")
    for team, score in zip(women_adj_teams, women_adj_scores):
        print(team, score)

