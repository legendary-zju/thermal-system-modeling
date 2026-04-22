from Aurora.tools.fluid_properties.wrappers import MoltenSaltWrapper


salt_engine = MoltenSaltWrapper("Solar Salt")
h = salt_engine.h_pT(1e5, 700)
T = salt_engine.T_ph(1e5, h)
p = salt_engine.p_hT(h, T)
d = salt_engine.d_ph(p, h)
msg = f"pressure: {p},  temperature: {T},  enthalpy: {h},  d: {d}"
print(msg)
