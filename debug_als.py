import pickle

with open("saved_models/hybrid/implicit_model.pkl", "rb") as f:
    state = pickle.load(f)

# Mock ALSImplicit object
from models.implicit.als_implicit import ALSImplicit
als = ALSImplicit()
als.load("saved_models/hybrid/implicit_model.pkl")

print("n_users:", len(als.user_to_idx), "n_items:", len(als.anime_to_idx))
print("Factors shape:", als.user_factors.shape, als.item_factors.shape)

try:
    res = als.get_similar_items(20, top_k=5)
    print("get_similar_items(20):", len(res))
    for r in res:
        print(" ", r)
except Exception as e:
    print("Error:", e)
