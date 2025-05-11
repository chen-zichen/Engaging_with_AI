import json
import os
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from itertools import combinations
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from statannotations.Annotator import Annotator
from constants import *

CONDITIONS = range(1, 7)

def get_data(question):
    file_path = PAIR + "/question_" + str(question) + "/info.json"
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

def get_ai_ans_only(question, condition):
    file_path = 'data/ai_outputs_diabetes/ans/gpt4_result_condition'
    if condition == 1 or condition == '1': file_path += '1.jsonl'
    if condition == 2 or condition == '2': file_path += '2.jsonl'
    if condition in [3, 4, 6] or condition in ['3','4','6']: file_path += '3.jsonl'
    if condition == 5 or condition == '5':
        if 1 <= question <= 20:
            file_path += '3.jsonl'
        if question > 20:
            file_path += '5.jsonl'
            question -= 20
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    return data[question-1]['gpt4_answer']

def get_ground_truth(question):
    file_path = 'data/ground_truth/diabetes_results.json'
    if question >= 21:
        file_path = 'data/ground_truth/diabetes_pre20_results.json'
    if question < 0:
        file_path = 'data/ground_truth/diabetes_attention_results.json'
    data = get_data(question)
    img1 = os.path.basename(data[0]["dish_id"])
    with open(file_path, 'r') as file:
        gts = json.load(file)
        if question < 0: # attention check questions
           gt = gts[question*(-1) - 1]['result']
        elif question >= 21: # pre survey
           gt = gts[question - 21]['result']
        else: # main questions
            gt = gts[question - 1]['result']
        if gt == img1: return '1'
        else: return '2'

def read_jsonl_files(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                return json_obj
            except json.JSONDecodeError:
                print("Error decoding JSON from:", line)

def load_data():
    data = pd.DataFrame()
    for i in CONDITIONS:
        for subdir, dirs, files in os.walk('./user_answers/condition' + str(i)):
            for file in files:
                if file[0] == '.': continue
                file_path = os.path.join(subdir, file)
                json_obj = read_jsonl_files(file_path)
                df = pd.DataFrame([json_obj])
                data = pd.concat([data, df], axis=0).reset_index(drop=True)
    
    data.columns = data.columns.str.replace("second", "2")
    data.columns = data.columns.str.replace("third", "3")
    for i in range(1, 21):
        if str(i) in data.columns:
            data.rename(columns={str(i): f"{i}_1"}, inplace=True)
    data = add_ai_gt(data)
    return data

def add_ai_gt(df):
    for i in range(1, 21):
        df[f"{i}_ai"] = None
        df[f"{i}_gt"] = None
    for index, row in df.iterrows():
        for i in range(1, 21):
            df.at[index, f'{i}_ai'] = get_ai_ans_only(question=i, condition=row['condition'])
            df.at[index, f'{i}_gt'] = "Meal " + get_ground_truth(question=i)
    return df

def user_summary(df, columns):
    filtered_df = df[df['user_id'] != "ai"]
    for col in columns:
        if filtered_df[col].dtype == object:
            print(f"{col}:")
            print(filtered_df[col].value_counts())
        else:
            print(f"{col}:")
            print(f"Min: {filtered_df[col].min()}; Max: {filtered_df[col].max()}")
            print(f"Mean: {filtered_df[col].mean()}")
            print(f"Median: {filtered_df[col].median()}")
            print(f"Standard Deviation: {filtered_df[col].std()}")
        print()

def plot_bar_plotly(df):
    means = df.mean()
    errors = df.sem()
    fig = go.Figure()

    for i, (column, mean) in enumerate(means.items()):
        fig.add_trace(go.Bar(
            name=column,
            x=[i],
            y=[mean],
            error_y=dict(type='data', array=[errors[column]]),
            width=0.5,
        ))
    fig.update_layout(
        title='Accuracy Comparison',
        xaxis=dict(title='Steps', tickmode='array', tickvals=list(range(len(means))), ticktext=means.index),
        yaxis=dict(title='Accuracy', range=[0, means.max() + errors.max() + 1]),
        showlegend=False,
        bargap=0.15,
    )
    fig.show()

def plot_unequal(data, title=None):
    df_data = pd.DataFrame.from_dict(data, orient='index').transpose()
    # f_val, p_val = stats.f_oneway(*[df_data[key] for key in data.keys()])
    keys = list(data.keys())
    if len(keys) != 2:
        raise ValueError("Wilcoxon signed-rank test can only compare two groups at a time.")

    # Wilcoxon signed-rank test between the first two groups in the dictionary
    group1, group2 = data[keys[0]], data[keys[1]]

    # checking if the data meets the assumptions of the test
    if len(group1) != len(group2):
        raise ValueError("Paired Wilcoxon test requires the same number of observations in both groups.")
    differences = np.array(group1) - np.array(group2)
    if np.all(differences == 0):
        raise ValueError("All differences between paired observations are zero, Wilcoxon test cannot be applied.")

    # # Histogram of differences
    # plt.figure(figsize=(8, 6))
    # sns.histplot(differences, kde=True)
    # plt.title('Distribution of Differences')
    # plt.xlabel('Difference (group1 - group2)')
    # plt.ylabel('Frequency')
    # plt.show()

    # Checking skewness (should be close to 0 for symmetric data)
    skewness = stats.skew(differences)
    print(f"Skewness of differences: {skewness:.4f}")

    # test
    stat, p_val = stats.wilcoxon(group1, group2)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_val}")
    # print(f"ANOVA F-value: {f_val}, p-value: {p_val}")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_data, palette="Set2")
    plt.title(title + f', p-value: {p_val:.4f}')
    plt.ylabel('Value')
    plt.xlabel('Groups')
    plt.show()

def plot_unequal_anova(data, title=None):
    df_data = pd.DataFrame.from_dict(data, orient='index').transpose()
    f_val, p_val = stats.f_oneway(*[df_data[key] for key in data.keys()])
    print(f"ANOVA F-value: {f_val}, p-value: {p_val}")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_data, palette="Set2")
    plt.title(title + f', p-value: {p_val:.4f}')
    plt.ylabel('Value')
    plt.xlabel('Groups')

    # plt.text(0.5, max(df_data.max()) + 0.1*max(df_data.max()), f'p-value: {p_val:.4f}', 
    #         ha='center', va='bottom', fontsize=12)
    plt.show()

def plot_unequal_multiple(data, title=None):
    # Convert the data to a DataFrame
    df_data = pd.DataFrame.from_dict(data, orient='index').transpose()

    # Extract the group data
    groups = [data[key] for key in data.keys()]
    keys = list(data.keys())

    # Perform the Kruskal-Wallis test
    stat, p_val = stats.kruskal(*groups)
    print(f"Kruskal-Wallis test statistic: {stat}, p-value: {p_val}")

    # If Kruskal-Wallis test is significant, perform post-hoc pairwise Mann-Whitney U tests
    if p_val < 0.05:
        print("\nPost-hoc pairwise Mann-Whitney U tests with Bonferroni correction:")
        pairwise_comparisons = list(combinations(keys, 2))
        p_values = []
        
        for group1, group2 in pairwise_comparisons:
            stat, p = stats.mannwhitneyu(data[group1], data[group2], alternative='two-sided')
            p_values.append((group1, group2, p))
            print(f"{group1} vs {group2} - test statistic: {stat}, p-value: {p}")

        # Apply Bonferroni correction for multiple comparisons
        corrected_p_values = [min(p[2] * len(pairwise_comparisons), 1.0) for p in p_values]

        print("\nCorrected p-values (Bonferroni):")
        for idx, corrected_p in enumerate(corrected_p_values):
            group1, group2 = pairwise_comparisons[idx]
            print(f"{group1} vs {group2} - corrected p-value: {corrected_p}")

    # Plotting the boxplot for all groups
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_data, palette="Set2")

    # Add titles and labels
    plt.title(f"{title}, Kruskal-Wallis p-value: {p_val:.4f}")
    plt.ylabel('Value')
    plt.xlabel('Groups')

    plt.show()

# def plot_unequal_multiple(data, title=None):
#     # Convert the data to a DataFrame
#     df_data = pd.DataFrame.from_dict(data, orient='index').transpose()

#     # Perform pairwise Wilcoxon signed-rank tests
#     keys = list(data.keys())
#     p_values = []
#     pairwise_comparisons = list(combinations(keys, 2))

#     print("Wilcoxon signed-rank test results:")
#     for group1, group2 in pairwise_comparisons:
#         stat, p_val = stats.wilcoxon(data[group1], data[group2])
#         p_values.append((group1, group2, p_val))
#         print(f"{group1} vs {group2} - test statistic: {stat}, p-value: {p_val}")

#     # Optional: Bonferroni correction for multiple comparisons
#     corrected_p_values = [min(p[2] * len(pairwise_comparisons), 1.0) for p in p_values]  # Bonferroni correction

#     print("\nCorrected p-values (Bonferroni):")
#     for idx, corrected_p in enumerate(corrected_p_values):
#         group1, group2 = pairwise_comparisons[idx]
#         print(f"{group1} vs {group2} - corrected p-value: {corrected_p}")

#     # Plotting the boxplot for all groups
#     plt.figure(figsize=(8, 6))
#     sns.boxplot(data=df_data, palette="Set2")

#     # Add titles and labels
#     plt.title(title, loc='center')
#     plt.ylabel('Value')
#     plt.xlabel('Groups')
#     plt.show()

def get_engagement(low, high):
    data = {
        'low': [],
        'high': []
    }

    for index, row in low.iterrows():
        count = 0
        for i in range(1, 21):
            if row[f'{i}_1'] != row[f'{i}_2']:
                count += 1
        data['low'].append(count/20)

    for index, row in high.iterrows():
        count = 0
        for i in range(1, 21):
            if row[f'{i}_1'] != row[f'{i}_2']:
                count += 1
        data['high'].append(count/20)

    return data

def get_step3_acc(low, high):
    data = {
        'low': [],
        'high': []
    }

    for index, row in low.iterrows():
        count = 0
        for i in range(1, 21):
            if row[f'{i}_3'] == row[f'{i}_gt']:
                count += 1
        data['low'].append(count)

    for index, row in high.iterrows():
        count = 0
        for i in range(1, 21):
            if row[f'{i}_3'] == row[f'{i}_gt']:
                count += 1
        data['high'].append(count)

    return data

def plot_bar(df, title=None):
    df_melted = df.melt(var_name='Step', value_name='Value')
    col_names = df.columns.tolist()
    
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df_melted, x='Step', y='Value', order=col_names, ci='sd', capsize=0.1)
    
    plt.ylabel('Value')
    plt.xlabel(None)
    plt.title(title)
    # plt.title('Comparison Across Steps')
    if len(df.columns) >= 4:
        plt.xticks(rotation=35, ha='right')
    else:
        plt.xticks(rotation=0, ha='center')
    sns.despine()
    
    f_statistic, p_value = stats.f_oneway(*[df[col] for col in df.columns])
    
    print(f"One-way ANOVA results:")
    print(f"F-statistic: {f_statistic}")
    print(f"p-value: {p_value}")
    
    tukey_results = pairwise_tukeyhsd(df_melted['Value'], df_melted['Step'])
    print("\nTukey's HSD test results:")
    print(tukey_results)
    
    y_max = df_melted['Value'].max() + df_melted['Value'].std()
    for group1, group2, meandiff, p_adj, lower, upper, reject in tukey_results.summary().data[1:]:
        if reject:
            x1 = col_names.index(group1)
            x2 = col_names.index(group2)
            y = y_max + 0.5 * abs(x1 - x2)
            plt.plot([x1, x1, x2, x2], [y, y+0.5, y+0.5, y], lw=1.5, c='k')
            plt.text((x1 + x2) * 0.5, y+0.5, "*", ha='center', va='bottom', color='k', fontsize=12)
            y_max = max(y_max, y + 1)
    # plt.ylim(df_melted['Value'].min() - 1, y_max + 1)
    plt.annotate(f"ANOVA P = {p_value:.5f}", xy=(0.5, -0.15), xycoords='axes fraction', 
                 ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_bar_wilcoxon(df, title=None):
    df_melted = df.melt(var_name='Step', value_name='Value')
    col_names = df.columns.tolist()
    
    plt.figure(figsize=(6, 6))
    sns.barplot(data=df_melted, x='Step', y='Value', order=col_names, ci='sd', capsize=0.1)
    plt.ylabel('Value')
    plt.xlabel(None)
    plt.title(title)
    
    if len(df.columns) >= 4:
        plt.xticks(rotation=35, ha='right')
    else:
        plt.xticks(rotation=0, ha='center')
    
    sns.despine()
    
    # Perform pairwise Wilcoxon signed-rank tests
    wilcoxon_results = []
    for col1, col2 in combinations(df.columns, 2):
        statistic, p_value = stats.wilcoxon(df[col1], df[col2])
        wilcoxon_results.append((col1, col2, statistic, p_value))
    
    # Print Wilcoxon test results
    print("Wilcoxon signed-rank test results:")
    for col1, col2, statistic, p_value in wilcoxon_results:
        print(f"{col1} vs {col2}: statistic = {statistic}, p-value = {p_value}")
    
    # Add significance markers to the plot
    y_max = df_melted['Value'].max() + df_melted['Value'].std()
    for col1, col2, _, p_value in wilcoxon_results:
        if p_value < 0.05:  # You can adjust this threshold
            x1 = col_names.index(col1)
            x2 = col_names.index(col2)
            y = y_max + 0.5 * abs(x1 - x2)
            plt.plot([x1, x1, x2, x2], [y, y+0.5, y+0.5, y], lw=1.5, c='k')
            plt.text((x1 + x2) * 0.5, y+0.5, "*", ha='center', va='bottom', color='k', fontsize=12)
            y_max = max(y_max, y + 1)
    
    plt.ylim(df_melted['Value'].min() - 1, y_max + 1)
    
    # Add overall p-value annotation (minimum p-value from all comparisons)
    min_p_value = min(result[3] for result in wilcoxon_results)
    plt.annotate(f"Min Wilcoxon p = {min_p_value:.5f}", xy=(0.5, -0.15), xycoords='axes fraction',
                 ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_parallel(df, ai_correct=None):
    if ai_correct is not None:
        df = df[df['ai_correct'] == ai_correct]
        df = df.drop('ai_correct', axis=1)
    
    # Create dimensions
    s1_dim = go.parcats.Dimension(values=df.s1_correct, label="Phase 1 Correctness", categoryarray=["Correct", "Incorrect"])
    s2_dim = go.parcats.Dimension(values=df.s2_correct, label="Phase 2 Correctness", categoryarray=["Correct", "Incorrect"])
    
    # Create parcats trace
    color = df['ai_correct'].astype('category').cat.codes
    colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]
    
    # Create the main parallel coordinates plot
    fig = go.Figure(data=[go.Parcats(
        dimensions=[s1_dim, s2_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', 
        hoverinfo='count+probability',
        labelfont={'color':'black', 'size': 18, 'family': 'Times'},
        tickfont={'color':'black', 'size': 16, 'family': 'Times'},
        arrangement='freeform'
    )])
    
    # Add dummy traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='AI Suggestion is Correct',
        line=dict(color='mediumseagreen', width=4),
        showlegend=True,
        xaxis='x2',
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name='AI Suggestion is Incorrect',
        line=dict(color='lightsteelblue', width=4),
        showlegend=True,
        xaxis='x2',
        yaxis='y2'
    ))
    
    # Update layout
    if ai_correct is not None:
        if ai_correct:
            fig.update_layout(title='AI CORRECT')
        else:
            fig.update_layout(title='AI INCORRECT')
    
    # Customize layout to remove background and position legend
    fig.update_layout(
        template=None,     
        showlegend=True,
        # paper_bgcolor='rgba(0,0,0,1)',  # Transparent background
        # plot_bgcolor='rgba(0,0,0,1)',   # Transparent plot area
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,  # Position legend outside the plot area
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        xaxis={'visible': False, 'showgrid': False},
        yaxis={'visible': False, 'showgrid': False},
        xaxis2={'visible': False, 'showgrid': False},
        yaxis2={'visible': False, 'showgrid': False},
        margin=dict(t=50, l=50, r=150)  # Add right margin to accommodate legend
    )
    
    config = {
        'toImageButtonOptions': {
            'format': 'svg',  # Vector format for highest quality
            'filename': 'parallel_coordinates',
            'height': 400,
            'width': 1000,
            'bgcolor': 'white',
            'scale': 2  # Increase the scale for higher resolution
        }
    }
    
    fig.show(config=config)

def plot_parallel_new(df):
    # Step 1: Create a new column for transitions
    # df['transition'] = df['s1_correct'] + " → " + str(df['ai_correct']) + " → " + df['s2_correct']
    df['transition'] = (
        df['s1_correct'] + " → " +
        df['ai_correct'].map({True: "AI Correct", False: "AI Incorrect"}) + " → " +
        df['s2_correct']
    )

    # Step 2: Define color mapping
    transition_categories = ["Correct → AI Correct → Correct",
                            "Correct → AI Incorrect → Correct",
                            "Correct → AI Correct → Incorrect",
                            "Correct → AI Incorrect → Incorrect",
                            "Incorrect → AI Correct → Correct",
                            "Incorrect → AI Incorrect → Correct",
                            "Incorrect → AI Correct → Incorrect",
                            "Incorrect → AI Incorrect → Incorrect"]
    
    # P1 correct  ---> AI correct ---> P2 correct
    color_map = {
        "Correct → AI Correct → Correct": "green",
        "Correct → AI Incorrect → Correct": "pink",
        "Correct → AI Correct → Incorrect": "red",
        "Correct → AI Incorrect → Incorrect": "red",
        "Incorrect → AI Correct → Correct": "orange",
        "Incorrect → AI Incorrect → Correct": "orange",
        "Incorrect → AI Correct → Incorrect": "blue",
        "Incorrect → AI Incorrect → Incorrect": "blue"
    }

    # Convert transition to categorical and get color codes
    df['transition'] = pd.Categorical(df['transition'], categories=transition_categories)
    color_codes = df['transition'].map(color_map)

    # Create dimensions
    s1_dim = go.parcats.Dimension(values=df.s1_correct, label="Phase 1 Correctness", categoryarray=[])
    s2_dim = go.parcats.Dimension(values=df.s2_correct, label="Phase 2 Correctness", categoryarray=[])

    # Create the parallel categories trace
    fig = go.Figure(data=[go.Parcats(
        dimensions=[s1_dim, s2_dim],
        line={'color': color_codes, 'shape': 'hspline'},
        hoveron='color',
        # hoverinfo='count+probability+dimension+category',
        hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform'
    )])

    # Add dummy traces for legend
    for trans, col in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=trans,
            line=dict(color=col, width=4),
            showlegend=True
        ))

    # Layout updates
    fig.update_layout(
        title='Transition Between Phases',
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        margin=dict(t=50, l=50, r=200),
        xaxis={'visible': False},
        yaxis={'visible': False}
    )

    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'parallel_coordinates',
            'height': 400,
            'width': 1000,
            'scale': 2
        }
    }

    fig.show(config=config)


def plot_parallel_three(df):
    # Step 1: Create transition type column
    df['transition'] = df['s1_correct'] + ' → ' + df['ai_correct'] + ' → ' + df['s2_correct']
    
    # Step 2: Assign color based on destination category (s2_correct)
    category_map = {
        "Correct → Correct → Correct": 0.0,
        "Correct → Incorrect → Correct": 0.33,
        "Correct → Correct → Incorrect": 0.66,
        "Correct → Incorrect → Incorrect": 1.0,
        "Incorrect → Correct → Correct": 0.0,
        "Incorrect → Incorrect → Correct": 0.33,
        "Incorrect → Correct → Incorrect": 0.66,
        "Incorrect → Incorrect → Incorrect": 1.0
    }

    # ANCHORED FROM THE BEGINNING
    # category_map = {
    #     "Correct → Correct → Correct": 0.0,
    #     "Correct → Incorrect → Correct": 0.33,
    #     "Correct → Correct → Incorrect": 0.0,
    #     "Correct → Incorrect → Incorrect": 0.33,
    #     "Incorrect → Correct → Correct": 0.66,
    #     "Incorrect → Incorrect → Correct": 1.0,
    #     "Incorrect → Correct → Incorrect": 0.66,
    #     "Incorrect → Incorrect → Incorrect": 1.0
    # }

    df['transition'] = df['s1_correct'] + ' → ' + df['ai_correct'] + ' → ' + df['s2_correct']
    df['dest_code'] = df['transition'].map(category_map)
    color = df['dest_code']

    # Step 3: Define a colorscale for destination categories
    # colorscale = [
    #     [0.0, 'rgba(26, 128, 187, 1)'],
    #     [0.5, 'rgba(26, 128, 187, 0.5)'],
    #     [1.0, 'rgba(234, 128, 28, 1)'],
    #     [1.5, 'rgba(234, 128, 28, 0.5)'],
    # ]
    colorscale = [
            [0.0, 'rgba(26, 128, 187, 1)'],      # Blue
            [0.33, 'rgba(26, 128, 187, 0.3)'],   # Blue faded
            [0.66, 'rgba(234, 128, 28, 1)'],     # Orange
            [1.0, 'rgba(234, 128, 28, 0.3)'],    # Orange faded
        ]

    # Create dimensions
    s1_dim = go.parcats.Dimension(values=df.s1_correct, label="Phase 1 Correctness", categoryarray=["Correct", "Incorrect"])
    s2_dim = go.parcats.Dimension(values=df.ai_correct, label="AI Correctness", categoryarray=["Correct", "Incorrect"])
    s3_dim = go.parcats.Dimension(values=df.s2_correct, label="Phase 2 Correctness", categoryarray=["Correct", "Incorrect"])

    # Create parallel categories plot
    fig = go.Figure(data=[go.Parcats(
        dimensions=[s1_dim, s2_dim, s3_dim],
        line={
                'color': color,              # still based on destination
                'colorscale': colorscale,
            },
        hoveron='color',
        hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform'
    )])

    # Final layout settings
    fig.update_layout(
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        xaxis={'visible': False, 'showgrid': False},
        yaxis={'visible': False, 'showgrid': False},
        xaxis2={'visible': False, 'showgrid': False},
        yaxis2={'visible': False, 'showgrid': False},
        margin=dict(t=50, l=50, r=200)
    )

    config = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'parallel_coordinates',
            'height': 400,
            'width': 1000,
            'scale': 2
        }
    }

    fig.show(config=config)



def plot_sankey(df):

    # Unique categories for nodes
    labels = ['S1 Correct', 'S1 Incorrect', 'AI Correct', 'AI Incorrect', 'S2 Correct', 'S2 Incorrect']
    label_map = {label: i for i, label in enumerate(labels)}

    # Count flows from S1 → AI
    df['s1_ai'] = df['s1_correct'] + '_' + df['ai_correct']
    s1_ai_counts = df.groupby(['s1_correct', 'ai_correct']).size().reset_index(name='count')

    # Count flows from AI → S2
    df['ai_s2'] = df['ai_correct'] + '_' + df['s2_correct']
    ai_s2_counts = df.groupby(['ai_correct', 's2_correct']).size().reset_index(name='count')

    destination_colors = {
        'S1 Correct': 'rgba(16, 128, 187, 0.6)',  
        'S1 Incorrect': 'rgba(241, 162, 38, 0.6)',
        'S2 Correct': 'rgba(16, 128, 187, 0.6)',  
        'S2 Incorrect': 'rgba(241, 162, 38, 0.6)',
        'AI Correct': 'rgba(242, 0, 0, 0.6)',     
        'AI Incorrect': 'rgba(41, 140, 140, 0.6)' 
    }
    # Sankey nodes and links
    nodes = dict(label=labels,
                 color=[destination_colors[i] for i in labels])


    # Links: S1 → AI
    links_s1_ai = {
        'source': [label_map['S1 ' + s1] for s1 in s1_ai_counts['s1_correct']],
        'target': [label_map['AI ' + ai] for ai in s1_ai_counts['ai_correct']],
        'value': s1_ai_counts['count'],
        'color': [destination_colors['AI ' + ai] for ai in s1_ai_counts['ai_correct']]
    }

    # Links: AI → S2
    links_ai_s2 = {
        'source': [label_map['AI ' + ai] for ai in ai_s2_counts['ai_correct']],
        'target': [label_map['S2 ' + s2] for s2 in ai_s2_counts['s2_correct']],
        'value': ai_s2_counts['count'],
        'color': [destination_colors['S2 ' + s2] for s2 in ai_s2_counts['s2_correct']]
    }

    # Combine links
    all_links = {
        'source': links_s1_ai['source'] + links_ai_s2['source'],
        'target': links_s1_ai['target'] + links_ai_s2['target'],
        'value': links_s1_ai['value'].tolist() + links_ai_s2['value'].tolist(),
        'color': links_s1_ai['color'] + links_ai_s2['color']
    }

    # Plot
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes['label']
        ),
        link=dict(
            source=all_links['source'],
            target=all_links['target'],
            value=all_links['value'],
            color=all_links['color']
        )
    )])

    fig.update_layout(title_text="Sankey Diagram: Phase Transitions", font_size=14)
    fig.show()