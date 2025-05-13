import json

with open("part1_qid2content_sol_avg_emb.json", "r") as f1, \
     open("part2_qid2content_sol_avg_emb.json", "r") as f2:
    
    part1 = json.load(f1)
    part2 = json.load(f2)

full = {**part1, **part2}

with open("qid2content_sol_avg_emb.json", "w") as fout:
    json.dump(full, fout)