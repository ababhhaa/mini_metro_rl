import pickle
with open("templates.pkl", "rb") as f:
    templates = pickle.load(f)
print(templates.keys())