import random
import math
import pandas as pd

L = 35
p = 5.1145
q = 6.1145

f_real = []
for i in range(3):
    f_real.append(random.random() * random.randint(-5, 5))

for i in range(3, L):
    f_real.append(p * f_real[i - 2] - q * f_real[i - 3])

a_real = []
for i in range(L):
    a_real.append(math.floor(f_real[i] + 0.5))

df = pd.DataFrame({"n": [_ for _ in range(L)], "a_seq": a_real})
df.to_csv("test.csv", index=False)
