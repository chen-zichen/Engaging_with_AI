import json
import os
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
# from statannotations.Annotator import Annotator
from constants import *

def get_correctness(ans, gt):
    if ans == gt: return True
    else: return False

def get_df(condition=1):
    df = pd.DataFrame()
    for subdir, dirs, files in os.walk('./user_answers/condition' + str(condition)):
        for file in files:
            if file[0].isdigit() == False: continue
            file_path = os.path.join(subdir, file)
            json_obj = read_jsonl_files(file_path)
            new_df = create_df(json_obj)
            df = pd.concat([df, new_df], ignore_index=True)
    
    return df

def plot_parallel_graph(df, title='AI Correct'):
    df = df[df['ai_correct'] == True]
    df['Color'] = df.apply(lambda row: 'All True' if all(row) else 'Any False', axis=1)
    # fig = px.parallel_categories(df, color=df['s1_correct'].astype('category').cat.codes, 
    #                          color_continuous_scale=px.colors.sequential.Plasma)
    fig = px.parallel_categories(df, color='Color')
    # fig = px.parallel_categories(df)
    fig.update_layout(title=title)
    fig.show()

def plot_test(df, ai_correct):
    df = df[df['ai_correct'] == ai_correct]
    df = df.drop('ai_correct', axis=1)
    # Create dimensions
    s1_dim = go.parcats.Dimension(values=df.s1_correct, label="s1 correctness", categoryarray=[True, False])
    s2_dim = go.parcats.Dimension(values=df.s2_correct, label="s2 correctness", categoryarray=[True, False])

    s3_dim = go.parcats.Dimension(
        values=df.s3_correct, label="s3 correctness", categoryarray=[True, False],
        ticktext=['True', 'False']
    )

    # Create parcats trace
    color = df['s3_correct'].astype('category').cat.codes
    colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]

    fig = go.Figure(data = [go.Parcats(dimensions=[s1_dim, s2_dim, s3_dim],
            line={'color': color, 'colorscale': colorscale},
            hoveron='color', hoverinfo='count+probability',
            labelfont={'size': 18, 'family': 'Times'},
            tickfont={'size': 16, 'family': 'Times'},
            arrangement='freeform')])
    if ai_correct:
        fig.update_layout(title='AI CORRECT')
    else:
        fig.update_layout(title='AI INCORRECT')
    fig.show()

def create_df(obj):
    data = {
        'ai_correct': [],
        's1_correct': [],
        's2_correct': [],
        's3_correct': []
    }

    for i in range(1, 21):
        gt = 'Meal ' + get_ground_truth(question=i)
        ai_ans = get_ai_ans_only(question=i, condition=int(obj['condition']))
        ai = get_correctness(ai_ans, gt)
        s1 = get_correctness(obj[str(i)], gt)
        s2 = get_correctness(obj[str(i) + '_second'], gt)
        s3 = get_correctness(obj[str(i) + '_third'], gt)

        if ai: data['ai_correct'].append(True)
        else: data['ai_correct'].append(False)

        if s1: data['s1_correct'].append(True)
        else: data['s1_correct'].append(False)

        if s2: data['s2_correct'].append(True)
        else: data['s2_correct'].append(False)

        if s3: data['s3_correct'].append(True)
        else: data['s3_correct'].append(False)

    df = pd.DataFrame(data)
    return df

def compare_answers(obj, compare, step=1):
    res = []
    c = obj['condition']
    for i in range(1, 21):
        if step == 1:
            user_ans = obj[str(i)]
        elif step == 2:
            user_ans = obj[str(i) + "_second"]
        elif step == 3:
            user_ans = obj[str(i) + "_third"]
        ai_ans = get_ai_ans_only(i, c)
        gt = 'Meal ' + get_ground_truth(question=i)
        if compare == 'ai_gt':
            if ai_ans == gt:
                res.append(True)
            else:
                res.append(False)
        elif compare == 'user_ai':
            if user_ans == ai_ans:
                res.append(True)
            else:
                res.append(False)
        elif compare == 'user_gt':
            if user_ans == gt:
                res.append(True)
            else:
                res.append(False)
    return res

def read_jsonl_files(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                return json_obj
            except json.JSONDecodeError:
                print("Error decoding JSON from:", line)

def count(res):
    same = 0
    diff = 0
    for i in res:
        if i == True:
            same += 1
        else:
            diff += 1
    print(f"Same: {same}, Different: {diff}")

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

def compare_data(compare, condition):
    # user_ai
    # user_gt
    # at_gt
    compare = 'user_gt'
    for subdir, dirs, files in os.walk('./user_answers/condition' + str(condition)):
        if compare == 'ai_gt':
            print("Comparing AI answers and ground truth")
        elif compare == 'user_ai':
            print("Comparing user answers and ai answers")
        elif compare == 'user_gt':
            print("Comparing user answers and ground truth")
        for file in files:
            file_path = os.path.join(subdir, file)
            json_obj = read_jsonl_files(file_path)
            res = compare_answers(json_obj, compare)
            count(res)

def load_condition_json(condition=1):
    data = {}
    for subdir, dirs, files in os.walk('./user_answers/condition' + str(condition)):
        for file in files:
            if file[0].isdigit() == False: continue
            file_path = os.path.join(subdir, file)
            json_obj = read_jsonl_files(file_path)
            data[json_obj['user_id']] = json_obj
    return data

def load_conditions(compare, conditions=range(1, 7), print_res=False):
    data = {}
    if type(conditions) == int: conditions = [conditions]
    for i in conditions:
        temp = []
        for subdir, dirs, files in os.walk('./user_answers/condition' + str(i)):
            for file in files:
                if file[0] == '.': continue
                file_path = os.path.join(subdir, file)
                json_obj = read_jsonl_files(file_path)
                res1 = compare_answers(json_obj, compare, step=1)
                res2 = compare_answers(json_obj, compare, step=2)
                res3 = compare_answers(json_obj, compare, step=3)
                if print_res:
                    print(f"User: {json_obj['user_id']}, comparing {compare}:")
                    print(f"Step 1: {res1.count(True)}")
                    print(f"Step 2: {res2.count(True)}")
                    print(f"Step 3: {res3.count(True)}")
                
                data[json_obj['user_id']] = [res1, res2, res3]
    return data

def load_user_change_ans(condition):
    data = {}
    for subdir, dirs, files in os.walk('./user_answers/condition' + str(condition)):
        for file in files:
            if file[0].isdigit() == False: continue
            file_path = os.path.join(subdir, file)
            json_obj = read_jsonl_files(file_path)
            step1 = load_ans_by_step(json_obj, step=1)
            step2 = load_ans_by_step(json_obj, step=2)
            step3 = load_ans_by_step(json_obj, step=3)

            step2_diff = []
            step3_diff = []
            for i in range(len(step1)):
                if step1[i] != step2[i]:
                    step2_diff.append(True)
                else:
                    step2_diff.append(False)
                
                if step1[i] != step3[i]:
                    step3_diff.append(True)
                else:
                    step3_diff.append(False)

            data[json_obj['user_id']] = [step2_diff, step3_diff]
            # data[json_obj['user_id']] = [step3_diff]
            
    return data

def load_ans_by_step(json_obj, step=1):
    res = []
    s = ''
    if step == 2: s += '_second'
    if step == 3: s += '_third'
    for i in range(1, 21):
        res.append(json_obj[str(i) + s])
    
    return res

def plot_bar_chart(data, title="User Answer Accuracies", x_lables=None):
    means = [np.mean(sample) for sample in data]
    std_devs = [np.std(sample) for sample in data]
    n_groups = len(data)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    bars = ax.bar(index, means, bar_width, yerr=std_devs, capsize=5)
    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(index)
    if x_lables == None:
        ax.set_xticklabels([f'Condition {i}' for i in range(1, n_groups+1)])
    else:
        ax.set_xticklabels(x_lables)
    filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_') + '.png'
    # plt.savefig(filename)
    plt.show()

def plot_two_bars_chart(compare1, compare2):
    data = load_conditions(compare1)

    means = [np.mean(sample) for sample in data]
    std_devs = [np.std(sample) for sample in data]
    n_groups = len(data)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35

    bars = ax.bar(index, means, bar_width, yerr=std_devs, capsize=5)

    data2 = load_conditions(compare2)

    means = [np.mean(sample) for sample in data2]
    std_devs = [np.std(sample) for sample in data2]
    n_groups = len(data2)
    index = np.arange(n_groups)
    bar_width = 0.35

    bars2 = ax.bar(index, means, bar_width, yerr=std_devs, capsize=5)

    title = ''
    # if compare1 == 'user_gt':
    #   title = "User Answer Accuracies"
    # if compare1 == 'user_ai':
    #   title = "User Follow AI Answers"
    # Add labels, title, and custom x-axis tick labels
    ax.set_xlabel('Group')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels([f'Condition {i}' for i in range(n_groups)])
    plt.show()

def plot_chart(compare):
  data = load_conditions(compare)

  means = [np.mean(sample) for sample in data]
  std_devs = [np.std(sample) for sample in data]
  n_groups = len(data)

  fig = go.Figure()

  fig.add_trace(
      go.Bar(
          x=[f'Condition {i}' for i in range(n_groups)],
          y=means,
          error_y=dict(
              type='data',
              array=std_devs,
              visible=True,
              thickness=1.5,
              width=1
          ),
          name=compare
      )
  )

  fig.update_layout(
      title=compare,
      xaxis_title='Group',
      yaxis_title='Values',
      legend_title='Legend'
  )

  fig.show()

def get_acc(data, condition, step=2):
    res = []
    for i in data.keys():
        correct = 0
        total = 0
        for i,q in enumerate(data[i][step]):
            total += 1
            gt = "Meal " + get_ground_truth(question=i+1)
            ai_ans = get_ai_ans_only(question=i+1, condition=condition)
            if ai_ans == gt and q:
                correct += 1
        res.append(correct/total)
    return res

def prepare_stats(data, data_len=3):
    res = [[] for _ in range(data_len)]
    for i in data.keys():
        for step in range(data_len):
            count = data[i][step].count(True)
            res[step].append(count)
    return res

def anova(data):
    f_statistic, p_value = stats.f_oneway(*data)
    print('F-statistic:', f'{f_statistic:.10f}')
    print('p-value:', f'{p_value:.10f}')

    if p_value <= 0.05:
        print("✅✅✅✅✅")

    return f_statistic, p_value

def posthoc(compare):
    data = load_conditions(compare)

    flat_data = [item for sublist in data for item in sublist]
    groups = []
    for i in range(0, 6):
      groups += [f'Condition {i}'] * len(data[0])
    f_statistic, p_value = stats.f_oneway(*data)
    posthoc = mc.MultiComparison(flat_data, groups)
    result = posthoc.tukeyhsd()

    print(result)

def t_test(compare):
    data = load_conditions(compare)

    for i in range(len(data)):
      for j in range(i + 1, len(data)):
          t_statistic, p_value = stats.ttest_ind(data[i], data[j])
          res = ''
          if p_value < 0.05: res = 'REJECT NULL'
          print(f'Comparison between Condition {i} and Group {j}: t-statistic = {t_statistic}, p-value = {p_value} {res}')

def plot_violin(compare):
    data = load_conditions(compare)

    def prepare_data_for_seaborn(data):
        flat_data = []
        for i, d in enumerate(data):
            for value in d:
                flat_data.append([f'Condition {i}', value])
        return pd.DataFrame(flat_data, columns=['Condition', 'Value'])

    df = prepare_data_for_seaborn(data)

    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x='Condition', y='Value', data=df, inner=None)
    sns.pointplot(x='Condition', y='Value', data=df, join=False, ci='sd', capsize=0.1, color='black')

    # Add statistical annotation
    pairs = [("Condition 0", "Condition 1"), ("Condition 0", "Condition 2"), ("Condition 0", "Condition 3"), ("Condition 0", "Condition 4"), ("Condition 0", "Condition 5")]
    annotator = Annotator(ax, pairs, data=df, x='Condition', y='Value')
    annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.title('User Answer Accuracy', y=1.4)
    plt.show()