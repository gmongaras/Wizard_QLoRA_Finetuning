import pandas as pd
import json
import random





# Percent of data to keep
per = 0.5






lines = []
idxs = dict()

files = [
    "data/comments_2009-10",
    "data/comments_2011-02",
    "data/comments_2012-02",
    "data/comments_2013-02",
    "data/comments_2014-02",
    "data/comments_2015-02",
    "data/comments_2016-11",
    "data/comments_2017-02",
    "data/comments_2018-02",
    "data/comments_2019-02",
]
for file in files:
    with open(file, "r") as f:
        for line in f:
            if random.random() > per:
                continue
            
            data = json.loads(line)
            if data["body"] != "[deleted]" and data["body"] != "[removed]":
                lines.append(data)
                idxs[data["id"]] = len(lines) - 1
    print(f"Finished reading {file}")
    
    
# Create dataframe
lines = pd.DataFrame(lines)


# Only negative comments
# lines = lines[lines["score"] < 0]


# Iterate over all lines, get the parent id and the text associated with it
# The output will be concatenated parent text and child text with two tabs in between
data = []
for index, row in lines.iterrows():
    # Only negative
    if row["score"] >= -10:
        continue
    
    parent_id = row["parent_id"][3:]
    if parent_id not in idxs:
        continue
    parent_text = lines.iloc[idxs[parent_id]]["body"].strip()
    if len(parent_text) == 0:
        continue
    data.append(f"{parent_text}\t\t{row['body'].strip()}")
    
    
# Make a hugginface dataset
from datasets import Dataset
data = Dataset.from_dict({"text": data})
print(f"Dataset size: {len(data)}")

# Save to hub
data.push_to_hub("gmongaras/reddit_negative")