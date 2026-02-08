"""Check implicit model structure."""
from models.implicit import ALSImplicit
from config import MODELS_DIR

print("Loading model...")
m = ALSImplicit()
m.load(MODELS_DIR / 'hybrid' / 'implicit_model.pkl')

print(f"user_factors shape: {m.user_factors.shape}")
print(f"item_factors shape: {m.item_factors.shape}")
print(f"user_to_idx count: {len(m.user_to_idx)}")
print(f"anime_to_idx count: {len(m.anime_to_idx)}")
print(f"idx_to_user count: {len(m.idx_to_user)}")
print(f"idx_to_anime count: {len(m.idx_to_anime)}")

# Check max indices
if m.user_to_idx:
    max_u_idx = max(m.user_to_idx.values())
    print(f"Max user index in mapping: {max_u_idx}")

if m.anime_to_idx:
    max_a_idx = max(m.anime_to_idx.values())
    print(f"Max anime index in mapping: {max_a_idx}")

# Sample keys
print(f"\nSample user_to_idx keys: {list(m.user_to_idx.keys())[:5]}")
print(f"Sample anime_to_idx keys: {list(m.anime_to_idx.keys())[:5]}")

# Test recommend_for_user with a user that exists
test_user = list(m.user_to_idx.keys())[0]
print(f"\nTesting recommend_for_user with user_id={test_user}")
try:
    recs = m.recommend_for_user(test_user, top_k=5)
    print(f"Got {len(recs)} recommendations:")
    for r in recs:
        print(f"  {r}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
