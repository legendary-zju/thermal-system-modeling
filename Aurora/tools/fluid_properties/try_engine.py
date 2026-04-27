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

# # 测试纯组分
# try:
#     # 联苯性质测试
#     T_bp_biphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'Biphenyl')
#     print(f"联苯在1atm下的沸点: {T_bp_biphenyl:.2f} K ({T_bp_biphenyl-273.15:.2f} °C)")
# except Exception as e:
#     print(f"联苯测试失败: {e}")
#
# try:
#     # 二苯醚性质测试
#     T_bp_diphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'DiphenylEther')
#     print(f"二苯醚在1atm下的沸点: {T_bp_diphenyl:.2f} K ({T_bp_diphenyl-273.15:.2f} °C)")
# except Exception as e:
#     print(f"二苯醚测试失败: {e}")
#
# # 测试预定义的Dowtherm A
# try:
#     T_bp_dowtherm = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'DowthermA')
#     print(f"Dowtherm A在1atm下的沸点: {T_bp_dowtherm:.2f} K ({T_bp_dowtherm-273.15:.2f} °C)")
# except Exception as e:
#     print(f"Dowtherm A测试失败: {e}")
#
# # 测试自定义混合物
# try:
#     fractions = [0.265, 0.735]  # 联苯, 二苯醚
#     T_custom = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'Biphenyl&DiphenylEther', fractions)
#     print(f"自定义混合物在1atm下的沸点: {T_custom:.2f} K ({T_custom-273.15:.2f} °C)")
# except Exception as e:
#     print(f"自定义混合物测试失败: {e}")
#
# # 获取所有可用流体
# fluids = CP.get_global_param_string("FluidsList").split(',')
#
# # 检查特定流体
# print("检查CoolProp 7.2.0中的流体支持:")
# print("=" * 50)
#
# # 检查联苯
# biphenyl_exists = 'Biphenyl' in fluids
# print(f"联苯(Biphenyl): {'✓ 支持' if biphenyl_exists else '✗ 不支持'}")
#
# # 检查二苯醚
# diphenyl_ether_exists = 'DiphenylEther' in fluids
# print(f"二苯醚(DiphenylEther): {'✓ 支持' if diphenyl_ether_exists else '✗ 不支持'}")
#
# # 检查Dowtherm A（预定义混合物）
# dowtherm_exists = 'DowthermA' in fluids
# print(f"Dowtherm A: {'✓ 支持' if dowtherm_exists else '✗ 不支持'}")
#
# # 列出所有相关的流体
# print("\n相关的流体列表:")
# for fluid in fluids:
#     if 'phenyl' in fluid.lower() or 'dowtherm' in fluid.lower():
#         print(f"  - {fluid}")
#
# # 尝试不同的名称
# possible_names = ['DiphenylEther', 'DiphenylOxide', 'PhenylEther']
# for name in possible_names:
#     if name in fluids:
#         print(f"找到二苯醚的别名: {name}")
#
# print(f"所有流体组分：{fluids}")



def check_refprop_integration():
    """检查CoolProp是否能访问REFPROP"""

    # 检查REFPROP版本
    try:
        refprop_version = CP.get_global_param_string("REFPROP_version")
        print(f"✓ CoolProp已集成REFPROP，版本: {refprop_version}")

        # 测试REFPROP流体
        try:
            # 测试联苯
            T_biphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'REFPROP::Biphenyl')
            print(f"✓ REFPROP联苯计算正常: {T_biphenyl:.2f} K")
        except Exception as e:
            print(f"✗ 联苯计算失败: {e}")

        # 测试二苯醚
        try:
            T_diphenyl = CP.PropsSI('T', 'P', 101325, 'Q', 0, 'REFPROP::DiphenylEther')
            print(f"✓ REFPROP二苯醚计算正常: {T_diphenyl:.2f} K")
        except Exception as e:
            print(f"✗ 二苯醚计算失败: {e}")

        return True
    except Exception as e:
        print(f"✗ CoolProp未检测到REFPROP: {e}")
        return False


check_refprop_integration()

print("获取REFPROP中所有流体列表...")
try:
    # 获取所有流体（可能很多）
    all_fluids = CP.get_global_param_string("REFPROP_fluid_list")
    fluids_list = all_fluids.split(',')

    print(f"REFPROP中有 {len(fluids_list)} 种流体")

    # 搜索包含'phenyl'或'biphenyl'的流体
    print("\n搜索包含'phenyl'或'biphenyl'的流体:")
    for fluid in fluids_list:
        fluid_lower = fluid.lower()
        if 'phenyl' in fluid_lower or 'biphenyl' in fluid_lower or 'diphenyl' in fluid_lower:
            print(f"  {fluid}")
except Exception as e:
    print(f"获取流体列表失败: {e}")