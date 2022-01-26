import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path')
args = parser.parse_args()

with open(args.path + "/utanet_dataset.pkl", 'rb') as f:
    utanet_dataset = pickle.load(f)
for i in utanet_dataset[0]['music_list']:
    print(i['曲名'])