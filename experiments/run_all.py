"""
运行所有预测模型
================
按顺序运行所有模型：数据准备 -> 各模型预测 -> 可视化

运行方式：
  python run_all_forecasts.py

可选参数：
  --skip-data-prep    跳过数据准备步骤
  --skip-baselines    跳过 baseline 模型
  --skip-deep-models  跳过深度学习模型 (Chronos, TimesFM, Toto)
  --skip-plot         跳过可视化
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd: list, cwd: str = None, env: dict = None) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'='*60}")
    print(f"运行: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(
        cmd,
        cwd=cwd or SCRIPT_DIR,
        env=env,
        capture_output=False,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='运行所有预测模型')
    parser.add_argument('--skip-data-prep', action='store_true', help='跳过数据准备')
    parser.add_argument('--skip-baselines', action='store_true', help='跳过 baseline 模型')
    parser.add_argument('--skip-deep-models', action='store_true', help='跳过深度学习模型')
    parser.add_argument('--skip-plot', action='store_true', help='跳过可视化')
    args = parser.parse_args()

    print("=" * 60)
    print("零样本电价预测 - 完整运行流程")
    print("=" * 60)

    success_count = 0
    total_count = 0

    # 1. 数据准备
    if not args.skip_data_prep:
        total_count += 1
        if run_command(["python", "../src/data_processing/prepare.py"]):
            success_count += 1
        else:
            print("❌ 数据准备失败")
            return

    # 2. Baseline 模型
    if not args.skip_baselines:
        total_count += 1
        # 使用 toto 虚拟环境运行 baseline（因为需要 sklearn）
        toto_python = os.path.join(SCRIPT_DIR, "../external/toto/.venv/bin/python")
        if os.path.exists(toto_python):
            if run_command([toto_python, "../src/models/baselines.py"]):
                success_count += 1
            else:
                print("⚠️ Baseline 模型运行失败")
        else:
            print(f"⚠️ 找不到 Python 解释器: {toto_python}")

    # 3. 深度学习模型
    if not args.skip_deep_models:
        # Chronos-2 (需要激活其虚拟环境)
        total_count += 1
        chronos_env = os.environ.copy()
        chronos_venv = os.path.join(SCRIPT_DIR, "../external/chronos-forecasting/.venv/bin/python")
        if os.path.exists(chronos_venv):
            if run_command([chronos_venv, "forecast_chronos2.py"],
                          cwd=os.path.join(SCRIPT_DIR, "../external/chronos-forecasting")):
                success_count += 1
            else:
                print("⚠️ Chronos-2 运行失败")
        else:
            print(f"⚠️ 找不到 Chronos 虚拟环境: {chronos_venv}")

        # TimesFM (需要激活其虚拟环境)
        total_count += 1
        timesfm_venv = os.path.join(SCRIPT_DIR, "../external/timesfm/.venv/bin/python")
        if os.path.exists(timesfm_venv):
            if run_command([timesfm_venv, "forecast_timesfm.py"],
                          cwd=os.path.join(SCRIPT_DIR, "../external/timesfm")):
                success_count += 1
            else:
                print("⚠️ TimesFM 运行失败")
        else:
            print(f"⚠️ 找不到 TimesFM 虚拟环境: {timesfm_venv}")

        # Toto (需要激活其虚拟环境)
        total_count += 1
        toto_venv = os.path.join(SCRIPT_DIR, "../external/toto/.venv/bin/python")
        if os.path.exists(toto_venv):
            if run_command([toto_venv, "forecast_toto.py"],
                          cwd=os.path.join(SCRIPT_DIR, "../external/toto")):
                success_count += 1
            else:
                print("⚠️ Toto 运行失败")
        else:
            print(f"⚠️ 找不到 Toto 虚拟环境: {toto_venv}")

    # 4. 可视化 & 表格生成
    if not args.skip_plot:
        total_count += 1
        toto_python = os.path.join(SCRIPT_DIR, "../external/toto/.venv/bin/python")
        if run_command([toto_python, "../src/evaluation/plotting.py"]):
            success_count += 1
        else:
            print("⚠️ 可视化失败")

        # 生成对比表格
        total_count += 1
        if run_command([toto_python, "../src/evaluation/tables.py"]):
            success_count += 1
        else:
            print("⚠️ 表格生成失败")

    # 总结
    print("\n" + "=" * 60)
    print("运行总结")
    print("=" * 60)
    print(f"成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("✅ 所有步骤成功完成！")
        print(f"\n输出文件（都在 ../data/results/ 目录）：")
        print(f"  📊 预测结果：")
        print(f"     - forecast_baselines.csv")
        print(f"     - forecast_toto.csv")
        print(f"     - forecast_timesfm.csv")
        print(f"     - forecast_chronos2.csv")
        print(f"  📈 可视化 & 表格：")
        print(f"     - forecast_comparison.png")
        print(f"     - forecast_metrics.csv")
        print(f"     - all_models_comparison.csv")
    else:
        print("⚠️ 部分步骤失败，请检查上面的输出")


if __name__ == "__main__":
    main()
