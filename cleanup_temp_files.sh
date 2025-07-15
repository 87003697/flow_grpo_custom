#!/bin/bash

echo "🧹 Hunyuan3D GRPO 临时文件清理工具"
echo "=================================="

# 检查文件大小
echo "📊 当前临时文件占用空间："
echo "  - Profiler日志: $(du -sh profiler_logs/ 2>/dev/null | cut -f1 || echo '0B')"
echo "  - 调试日志: $(du -ch *.log 2>/dev/null | tail -1 | cut -f1 || echo '0B')"
echo "  - 检查点文件: $(du -sh checkpoints/ 2>/dev/null | cut -f1 || echo '0B')"
echo ""

# 1. 清理Profiler日志 (最大)
if [ -d "profiler_logs" ]; then
    read -p "🔴 删除Profiler日志目录 ($(du -sh profiler_logs/ | cut -f1))? [y/N]: " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        rm -rf profiler_logs/
        echo "✅ 已删除 profiler_logs/"
    fi
fi

# 2. 清理调试日志
if ls *.log 1> /dev/null 2>&1; then
    read -p "🟡 删除所有.log调试文件 ($(ls *.log | wc -l)个文件)? [y/N]: " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        rm -f *.log
        echo "✅ 已删除所有.log文件"
    fi
fi

# 3. 清理临时Python文件
temp_files=(
    "flow_grpo/diffusers_patch/hunyuan3d_pipeline_with_logprob.py.bak"
    "flow_grpo/trainer_3d_simplified_backup.py"
    "flow_grpo/trainer_3d_simplified_new.py"
    "flow_grpo/trainer_3d_ultra_simplified.py"
    "scripts/single_node/run_memory_optimized.sh"
    "scripts/train_hunyuan3d_ultra_simple.py"
    "temp_fix.py"
    "trainer_3d_simplified.py"
)

echo ""
echo "🟡 临时Python文件:"
for file in "${temp_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

read -p "删除这些临时Python文件? [y/N]: " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    for file in "${temp_files[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "✅ 已删除 $file"
        fi
    done
fi

# 4. 检查点文件 (谨慎)
echo ""
echo "🟢 检查点文件目录:"
if [ -d "checkpoints" ]; then
    du -sh checkpoints/*/
    echo ""
    echo "⚠️  注意：检查点包含训练的模型权重，删除前请确认不需要："
    echo "  - hunyuan3d_grpo_simplified: 主要训练检查点"
    echo "  - 其他目录: 测试检查点"
    echo ""
    read -p "是否要删除测试检查点目录 (保留hunyuan3d_grpo_simplified)? [y/N]: " confirm
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        for dir in checkpoints/*/; do
            dirname=$(basename "$dir")
            if [[ "$dirname" != "hunyuan3d_grpo_simplified" ]]; then
                rm -rf "$dir"
                echo "✅ 已删除 $dir"
            fi
        done
    fi
fi

echo ""
echo "🎉 清理完成！当前磁盘使用情况："
df -h . | tail -1 