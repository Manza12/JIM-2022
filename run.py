from tqdm import tqdm
from extraction import extract_harmonic, extract_inharmonic

for i in tqdm(range(21, 109)):
    name = str(i)
    extract_harmonic(name)
    extract_inharmonic(name)
