CONDITION = 1
PRE_SURVEY_Q = 1

IMG = "data/images"
PAIR = "data/diabetes-pair"
PRE_PAIR = "data/pre_pair"
SEGMENTS = "data/seg_images"
P1_DONE_TEXT = "You've completed the first part. Now, let's move on to the next part."
P2_DONE_TEXT = "Great! Let's move on to the third part."
AI_SUGGESTION = "Now, we're showing you what the AI suggests."

Q_MAP = {1: 2, 2: 8, 3: 16, 4: 11, 5: 13, 6: 7, 7: 5, 8: 17, 9: 14, 10: 10, 
         11: 6, 12: 1, 13: 18, 14: 19, 15: 20, 16: 9, 17: 15, 18: 4, 19: 3, 20: 12,
         21: 22, 22: 35, 23: 26, 24: 39, 25: 30, 26: 32, 27: 21, 28: 27, 29: 25, 30: 24, 
         31: 23, 32: 34, 33: 28, 34: 29, 35: 36, 36: 31, 37: 40, 38: 33, 39: 38, 40: 37}
CONDITION5_FIRST_MAP = {1: 35, 2: 28, 3: 21, 4: 29, 5: 22}

AI_KNOELEDGE = {
    "No Knowledge": None,
    "I have used AI systems such as chatgpt, midjourney etc.": None,
    "I understand basic AI concepts": None,
    "I understand advanced AI concepts": None,
    "I am an AI researcher": None,
    "I am an AI developer": None
}
AI_USED = {
    "ChatGPT": None,
    "Claude.ai": None,
    "Perplexity": None,
    "Pi": None,
    "MidJourney": None,
}
AI_PURPOSE = {
    "Writing assistance and editing": None,
    "Problem-solving and brainstorming": None,
    "Language translation and explanation": None,
    "Math and coding": None,
    "Summarizing long texts or articles": None,
    "Explaining complex concepts in simpler terms": None,
    "Answering questions": None,
    "Providing general advice": None,
    "Help with making decisions": None,
    "Image generation": None,
    "Video generation": None,
}
PRE_SURVEY = [
    'pre_frequency',
    'pre_satisfaction',
    'pre_complexity',
    'pre_understand',
    'pre_reliability',
    'pre_trust',
    'pre_accurate',
    'pre_future',
    'pre_use_others',
    'pre_purpose_others'
]
POST_SURVEY = [
    'post_condition_satisfaction',
    'post_condition_satisfaction_2',
    'post_condition_text',
    'post_satisfaction',
    'post_complexity',
    'post_reliability',
    'post_accurate',
    'post_base_question',
    'post_challenge',
    'post_open',
    'post_use',
    'post_confi',
    'post_understand_condition',
    'post_consistent',
    'post_demanding',
    'post_easier',
    'post_mental_demanding',
    'post_autonomy',
    'post_trust',
    'post_confident',
    'post_notice',
    'post_help_final',
    'post_increase_trust',
    'post_comfortable',
    'post_like_use',
    'post_trust_change'
]
CONDITION_KEY_WORD = {
    1: "visual explanation",
    2: "AI's explanation",
    3: "confidence levels",
    4: "inputting the confidence levels",
    5: "bar chart",
    6: "AI's questions"
}

PROLIFIC = {
    0: ['https://app.prolific.com/submissions/complete?cc=CYPC4571', 'CYPC4571'],
    1: ['https://app.prolific.com/submissions/complete?cc=CK50C5XQ', 'CK50C5XQ'],
    2: ['https://app.prolific.com/submissions/complete?cc=C14Y0R8N', 'C14Y0R8N'],
    3: ['https://app.prolific.com/submissions/complete?cc=CQPF433R', 'CQPF433R'],
    4: ['https://app.prolific.com/submissions/complete?cc=COYKXGDM', 'COYKXGDM'],
    5: ['https://app.prolific.com/submissions/complete?cc=C1AJKMFH', 'C1AJKMFH'],
}