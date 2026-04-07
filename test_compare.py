import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models.hybrid.hybrid_engine import HybridEngine
he = HybridEngine()
he.load('saved_models/hybrid')
print('Weights:', he.weights)
for method in ['content', 'collaborative', 'implicit', 'hybrid']:
    try:
        recs = he.recommend_similar_anime(20, top_k=5, method=method)
        print('--- Method:', method, ', count:', len(recs), '---')
        if recs:
            for r in recs[:2]: 
                print(r['mal_id'])
    except Exception as e:
        print('Method', method, 'failed:', e)
