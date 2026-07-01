# - ori: repeat (copying, respecting the original text)
# - num: digit (number recognition, counting)
# - order: order (sequence, precedence, spatial relationships)
# - recognition: corresponding recognition
# - format: format/rewrite (additions, deletions, modifications at the word/character level, string output requirements with varying precision)
# - logit: logic (making precise assumptions, executing different instructions based on varying conditions)
# - reco: active attention/cognition/distinction (following instructions to focus more on certain parts or ignore others, maintaining attention integrity) recognition


sc_map = {
    "list-single_query_id": {
        'format': 'format',
        "correct": 'recog',
        'ori': 'ori',
        # 'num': 'format',
        },
    "list-multi_query_id": {
        "format": 'format',
        "num": 'num',
        "correct": 'recog',
        "order": 'spat',
        },
    "list-offset_query_id": {
        'format': 'format',
        "correct": ['recog', 'spat'],
        'ori': 'ori',
        },

    "list-offset_query_element": {
        'format': 'format',
        "correct": 'spat',
        'ori': 'ori',
        },
    "list-blur_offset_query_id": {
        'ori': 'ori',
        'position': 'spat',
        'format': 'format',
        },
    "list-blur_offset_query_element": {
        'ori': 'ori',
        'position': 'spat',
        'format': 'format',
        },
    "multidoc-batch_label": {
        'candi_ori': 'ori',
        'logit_correct': 'logit',
        'num_doc': ['num', 'recog'],
        'format': 'format',
        },
    "multidoc-find_dup_text": {
        'correct': ['logit', 'recog'],
        'num_text': ['num', 'logit'],
        'ori': 'ori',
        'format': 'format',
        },
    "onedoc-extract": {
        'format': 'format',
        "correct": 'recog',
        "order": 'spat',
        "ori": 'ori',
        },
    "onedoc-qa": {
        "correct": 'logit',
        'format': 'format',
        },
    "onedoc-repeat": {
        'format': 'format',
        'ori': 'ori',
        'hit': 'recog',
        'num': 'num',
        "correct": 'logit',
        },
    }
