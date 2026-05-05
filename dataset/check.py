
import pandas as pd

meta_train = pd.read_csv("C:\\Users\\Asus\\Desktop\\progetto_manutenzione\\dataset\\dataset_new\\arc_dataset_meta_train.csv")
meta_test  = pd.read_csv("C:\\Users\\Asus\\Desktop\\progetto_manutenzione\\dataset\\dataset_new\\arc_dataset_meta_test.csv")

files_train = set(meta_train["filename"].str.replace("_Study00[12]_Raw Data.mat", "", regex=True))
files_test  = set(meta_test["filename"].str.replace("_Study00[12]_Raw Data.mat", "", regex=True))

overlap = files_train & files_test
print(f"File sorgente in comune: {len(overlap)}")
print(f"File train: {len(files_train)}")
print(f"File test:  {len(files_test)}")
if overlap:
    print("LEAKAGE - file in comune:")
    for f in sorted(overlap):
        print(f"  {f}")
else:
    print("✅ Nessun file sorgente condiviso — split corretto")