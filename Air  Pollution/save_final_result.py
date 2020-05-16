"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande
Description: 
Concepts Used:

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

import glob
import pandas as pd
import sys

# Final finals needed to be converted to a CSV for visualization
finalFiles = sys.argv[1]
paths = glob.glob(finalFiles)

x = []
for path in paths:
  with open(path) as f:
      for line in f:
        x.append(eval(line))
db = pd.DataFrame(x)

db.to_csv('pollutant.csv')