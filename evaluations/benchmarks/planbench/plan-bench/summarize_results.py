import json, os

results_dir = 'results/blocksworld'
for root, dirs, files in os.walk(results_dir):
    for f in sorted(files):
        if not f.endswith('.json'):
            continue
        path = os.path.join(root, f)
        rel = os.path.relpath(path, results_dir)
        with open(path) as fh:
            data = json.load(fh)
        instances = data.get('instances', [])
        total = len(instances)
        has_resp = sum(1 for i in instances if i.get('llm_raw_response'))

        # t3 (verification) uses llm_correct_binary/w_type/w_expl instead of llm_correct
        if 'verification' in f:
            binary = sum(1 for i in instances if i.get('llm_correct_binary'))
            w_type = sum(1 for i in instances if i.get('llm_correct_w_type'))
            w_expl = sum(1 for i in instances if i.get('llm_correct_w_expl'))
            evaluated = sum(1 for i in instances if i.get('llm_correct_binary') is not None)
            print(f'{rel}: binary={binary} w_type={w_type} w_expl={w_expl} / {evaluated} evaluated  ({has_resp} had responses)')
        else:
            correct = sum(1 for i in instances if i.get('llm_correct'))
            print(f'{rel}: {correct}/{total} correct  ({has_resp} had responses)')
