import pandas as pd, numpy as np
d=pd.read_csv('results/PD_HD_validation/exploratory/CCS_profile_real_vs_fake.csv')
f=d[['N','CCS_PD_FAKE','p_PD_FAKE']].copy()
# local maxima
v=f.CCS_PD_FAKE.values
loc=[i for i in range(1,len(v)-1) if v[i]>=v[i-1] and v[i]>=v[i+1]]
print("CHERRY-PICKED 17 — local maxima in the CCS profile:")
print("%5s %10s %10s" % ("N","CCS","p_sibling"))
for i in loc:
    star=" <== sig" if f.p_PD_FAKE.iloc[i]<0.05 else ""
    print("%5d %10.4f %10.4f%s" % (f.N.iloc[i],v[i],f.p_PD_FAKE.iloc[i],star))
print("\nall sizes where the fake set is significant (p<0.05):")
print(f[f.p_PD_FAKE<0.05].to_string(index=False))
print("\ntop 6 CCS values overall:")
print(f.nlargest(6,'CCS_PD_FAKE').to_string(index=False))
