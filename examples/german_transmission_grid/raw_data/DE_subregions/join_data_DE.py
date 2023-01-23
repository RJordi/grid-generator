import json


files = [f'./substations/substations_DE_18072016_{num}.json' for num in range(1,17)]


def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('substations_total.json', 'w') as output_file:
        json.dump(result, output_file)

merge_JsonFiles(files)