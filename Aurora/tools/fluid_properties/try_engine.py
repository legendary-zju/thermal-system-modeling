from Aurora.tools.fluid_properties.wrappers import ThermalConductingMediumWrapper

# this file is created just for testing fluid calculating engine.
# salt_engine = MoltenSaltWrapper("Solar Salt")
# h = salt_engine.h_pT(1e5, 700)
# T = salt_engine.T_ph(1e5, h)
# p = salt_engine.p_hT(h, T)
# d = salt_engine.d_ph(p, h)
# s = salt_engine.s_ph(p, h)
# msg = f"pressure: {p},  temperature: {T},  enthalpy: {h},  density: {d},   entropy: {s}"
# print(msg)

import CoolProp.CoolProp as CP

# 测试纯组分
try:
    # 联苯性质测试
    T_bp_biphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'Biphenyl')
    print(f"联苯在1atm下的沸点: {T_bp_biphenyl:.2f} K ({T_bp_biphenyl-273.15:.2f} °C)")
except Exception as e:
    print(f"联苯测试失败: {e}")

try:
    # 二苯醚性质测试
    T_bp_diphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'DiphenylEther')
    print(f"二苯醚在1atm下的沸点: {T_bp_diphenyl:.2f} K ({T_bp_diphenyl-273.15:.2f} °C)")
except Exception as e:
    print(f"二苯醚测试失败: {e}")

# 测试预定义的Dowtherm A
try:
    T_bp_dowtherm = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'DowthermA')
    print(f"Dowtherm A在1atm下的沸点: {T_bp_dowtherm:.2f} K ({T_bp_dowtherm-273.15:.2f} °C)")
except Exception as e:
    print(f"Dowtherm A测试失败: {e}")

# 测试自定义混合物
try:
    fractions = [0.265, 0.735]  # 联苯, 二苯醚
    T_custom = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'Biphenyl&DiphenylEther', fractions)
    print(f"自定义混合物在1atm下的沸点: {T_custom:.2f} K ({T_custom-273.15:.2f} °C)")
except Exception as e:
    print(f"自定义混合物测试失败: {e}")

# 获取所有可用流体
fluids = CP.get_global_param_string("FluidsList").split(',')

# 检查特定流体
print("检查CoolProp 7.2.0中的流体支持:")
print("=" * 50)

# 检查联苯
biphenyl_exists = 'Biphenyl' in fluids
print(f"联苯(Biphenyl): {'✓ 支持' if biphenyl_exists else '✗ 不支持'}")

# 检查二苯醚
diphenyl_ether_exists = 'DiphenylEther' in fluids
print(f"二苯醚(DiphenylEther): {'✓ 支持' if diphenyl_ether_exists else '✗ 不支持'}")

# 检查Dowtherm A（预定义混合物）
dowtherm_exists = 'DowthermA' in fluids
print(f"Dowtherm A: {'✓ 支持' if dowtherm_exists else '✗ 不支持'}")

# 列出所有相关的流体
print("\n相关的流体列表:")
for fluid in fluids:
    if 'phenyl' in fluid.lower() or 'dowtherm' in fluid.lower():
        print(f"  - {fluid}")

# 尝试不同的名称
possible_names = ['DiphenylEther', 'DiphenylOxide', 'PhenylEther']
for name in possible_names:
    if name in fluids:
        print(f"找到二苯醚的别名: {name}")

print(f"所有流体组分：{fluids}")
