#!/usr/bin/env python3
"""
测试 Image-Mesh 配对数据的语义一致性评分
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
from kiui.mesh import Mesh
import time
import argparse
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reward_models.uni3d_scorer import Uni3DScorer


def load_image_as_tensor(image_path, device="cuda"):
    """将图像加载为 CLIP 预处理的张量"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # CLIP 预处理 (根据 open_clip 的标准预处理)
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        image_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)
        return image_tensor.to(device)
        
    except Exception as e:
        print(f"❌ 加载图像 {image_path} 失败: {e}")
        return None


def load_mesh_as_kiui(mesh_path):
    """将 .glb 文件加载为 kiui mesh"""
    try:
        mesh = Mesh.load(str(mesh_path))
        return mesh
        
    except Exception as e:
        print(f"❌ 加载 mesh {mesh_path} 失败: {e}")
        return None


def find_image_mesh_pairs(dataset_root):
    """查找 image-mesh 配对数据"""
    meshes_dir = dataset_root / "meshes"
    images_dir = dataset_root / "images"
    
    pairs = []
    
    # 遍历所有图像文件
    for image_path in images_dir.glob("*.png"):
        name = image_path.stem  # 获取文件名（不含扩展名）
        
        # 查找对应的 mesh 文件
        mesh_path = meshes_dir / f"{name}_textured_frame_000000.glb"
        
        if mesh_path.exists():
            pairs.append({
                'name': name,
                'image_path': image_path,
                'mesh_path': mesh_path
            })
        else:
            print(f"⚠️ 未找到对应的 mesh 文件: {mesh_path}")
    
    return pairs


def compute_recall_at_k(scorer, pairs, device, k_values=[1, 5, 10]):
    """
    计算 Recall@K 指标
    
    Args:
        scorer: Uni3D 评分器
        pairs: image-mesh 配对数据
        device: 计算设备
        k_values: 要计算的 K 值列表
    
    Returns:
        dict: 包含 recall@k 结果的字典
    """
    print(f"\n🔍 开始计算 Recall@K 指标...")
    
    # 预加载所有 mesh
    print("🔄 正在预加载所有 mesh...")
    all_meshes = {}
    mesh_features = {}
    
    for pair in pairs:
        mesh = load_mesh_as_kiui(pair['mesh_path'])
        if mesh is not None:
            all_meshes[pair['name']] = mesh
            print(f"  ✅ 加载 mesh: {pair['name']}")
    
    # 计算所有 mesh 的特征
    print("🔄 正在计算所有 mesh 的特征...")
    for name, mesh in all_meshes.items():
        try:
            # 将 mesh 转换为点云
            from reward_models.uni3d_scorer.utils.processing import prepare_pointcloud_batch
            pointcloud_batch = prepare_pointcloud_batch([mesh], num_points=8192)
            pointcloud_batch = pointcloud_batch.to(device)
            
            # 使用 Uni3D 编码点云 - 直接输出 CLIP 嵌入空间的特征
            with torch.no_grad():
                pc_features = scorer.uni3d_model.encode_pc(pointcloud_batch)  # 已经是 CLIP 空间的特征
                pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                mesh_features[name] = pc_features  # 保持在 GPU 上
            
            print(f"  ✅ 计算特征: {name} (维度: {pc_features.shape})")
            
        except Exception as e:
            print(f"  ❌ 计算特征失败: {name} - {e}")
            continue
    
    # 对每个图像计算与所有 mesh 的相似度
    recall_results = {k: [] for k in k_values}
    detailed_results = []
    
    print("\n🧪 开始计算相似度矩阵...")
    
    for i, pair in enumerate(pairs):
        if pair['name'] not in mesh_features:
            continue
            
        print(f"\n📊 [{i+1}/{len(pairs)}] 处理图像: {pair['name']}")
        
        # 加载图像
        image_tensor = load_image_as_tensor(pair['image_path'], device)
        if image_tensor is None:
            continue
        
        # 计算图像特征
        try:
            with torch.no_grad():
                image_features = scorer.clip_model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                print(f"  📷 图像特征维度: {image_features.shape}")
                
        except Exception as e:
            print(f"  ❌ 计算图像特征失败: {e}")
            continue
        
        # 计算与所有 mesh 的相似度
        similarities = []
        for mesh_name, mesh_feat in mesh_features.items():
            # 确保两个特征都在同一设备上，且维度匹配
            if image_features.shape[-1] != mesh_feat.shape[-1]:
                print(f"  ⚠️ 维度不匹配：图像 {image_features.shape} vs mesh {mesh_feat.shape}")
                print(f"  ℹ️ 这表明 Uni3D 的 trans2embed 层可能没有正确加载")
                continue
                
            similarity = torch.mm(image_features, mesh_feat.T).item()
            similarities.append({
                'mesh_name': mesh_name,
                'similarity': similarity
            })
        
        if not similarities:
            print(f"  ❌ 无法计算相似度，跳过: {pair['name']}")
            continue
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 找到正确匹配的排名
        correct_match_rank = None
        for rank, sim_result in enumerate(similarities, 1):
            if sim_result['mesh_name'] == pair['name']:
                correct_match_rank = rank
                break
        
        if correct_match_rank is None:
            print(f"  ❌ 未找到正确匹配: {pair['name']}")
            continue
        
        # 计算 recall@k
        for k in k_values:
            if correct_match_rank <= k:
                recall_results[k].append(1)
            else:
                recall_results[k].append(0)
        
        # 保存详细结果
        top_5_matches = similarities[:5]
        detailed_results.append({
            'image_name': pair['name'],
            'correct_rank': correct_match_rank,
            'top_5_matches': top_5_matches,
            'correct_similarity': similarities[correct_match_rank-1]['similarity']
        })
        
        print(f"  ✅ 正确匹配排名: {correct_match_rank}")
        print(f"  📊 Top 3 匹配:")
        for rank, sim in enumerate(similarities[:3], 1):
            status = "✅" if sim['mesh_name'] == pair['name'] else "❌"
            print(f"    {rank}. {sim['mesh_name']}: {sim['similarity']:.4f} {status}")
    
    # 计算最终 recall@k
    final_recall = {}
    for k in k_values:
        if recall_results[k]:
            final_recall[f'recall@{k}'] = np.mean(recall_results[k])
        else:
            final_recall[f'recall@{k}'] = 0.0
    
    return final_recall, detailed_results


def test_image_mesh_pairs(num_test=None, save_results=False):
    """测试 Image-Mesh 配对数据"""
    print("🚀 开始测试 Image-Mesh 配对数据的语义一致性...")
    
    # 数据集路径
    dataset_root = project_root / "dataset" / "eval3d"
    
    # 查找配对数据
    print("🔍 正在查找 image-mesh 配对数据...")
    pairs = find_image_mesh_pairs(dataset_root)
    print(f"📁 找到 {len(pairs)} 个配对数据")
    
    if len(pairs) == 0:
        print("❌ 没有找到配对数据")
        return False
    
    # 初始化评分器
    print("🔄 正在初始化 Uni3D 评分器...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = Uni3DScorer(device=device, dtype=torch.float32)
        print("✅ Uni3D 评分器初始化成功")
    except Exception as e:
        print(f"❌ Uni3D 评分器初始化失败: {e}")
        return False
    
    # 选择测试的配对数量
    if num_test is None:
        test_pairs = pairs
    else:
        test_pairs = pairs[:num_test]
    
    print(f"🧪 测试 {len(test_pairs)} 个配对...")
    
    results = []
    total_time = 0
    
    for i, pair in enumerate(test_pairs):
        print(f"\n📊 [{i+1}/{len(test_pairs)}] 测试: {pair['name']}")
        
        # 加载图像
        image_tensor = load_image_as_tensor(pair['image_path'], device)
        if image_tensor is None:
            continue
            
        # 加载 mesh
        mesh = load_mesh_as_kiui(pair['mesh_path'])
        if mesh is None:
            continue
        
        print(f"  📷 图像尺寸: {image_tensor.shape}")
        print(f"  🔺 Mesh: {mesh.v.shape[0]} 顶点, {mesh.f.shape[0]} 面")
        
        # 计算语义相似度
        try:
            start_time = time.time()
            score = scorer._compute_image_semantic_score(mesh, image_tensor, num_points=8192)
            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            
            print(f"  ✅ 语义相似度评分: {score:.4f} (耗时: {processing_time:.2f}s)")
            
            results.append({
                'name': pair['name'],
                'score': score,
                'image_path': str(pair['image_path']),
                'mesh_path': str(pair['mesh_path']),
                'processing_time': processing_time,
                'mesh_vertices': mesh.v.shape[0],
                'mesh_faces': mesh.f.shape[0]
            })
            
        except Exception as e:
            print(f"  ❌ 评分失败: {e}")
            continue
    
    # 生成报告
    if results:
        print("\n" + "="*100)
        print("📊 Image-Mesh 语义一致性评分报告")
        print("="*100)
        
        # 按评分排序
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print(f"{'排名':<4} {'名称':<25} {'评分':<8} {'时间':<8} {'顶点':<8} {'面':<8} {'文件大小'}")
        print("-" * 100)
        
        for rank, result in enumerate(results_sorted, 1):
            name = result['name'][:24]  # 截断长名称
            score = result['score']
            proc_time = result['processing_time']
            vertices = result['mesh_vertices']
            faces = result['mesh_faces']
            
            # 获取文件大小
            image_size = Path(result['image_path']).stat().st_size / 1024  # KB
            mesh_size = Path(result['mesh_path']).stat().st_size / 1024   # KB
            
            print(f"{rank:<4} {name:<25} {score:<8.4f} {proc_time:<8.2f} {vertices:<8} {faces:<8} 图像:{image_size:.0f}KB, Mesh:{mesh_size:.0f}KB")
        
        # 统计信息
        scores = [r['score'] for r in results]
        times = [r['processing_time'] for r in results]
        
        print("\n📈 统计信息:")
        print(f"  平均评分: {np.mean(scores):.4f}")
        print(f"  标准差: {np.std(scores):.4f}")
        print(f"  最高评分: {np.max(scores):.4f}")
        print(f"  最低评分: {np.min(scores):.4f}")
        print(f"  评分范围: {np.max(scores) - np.min(scores):.4f}")
        print(f"  平均处理时间: {np.mean(times):.2f}s")
        print(f"  总处理时间: {total_time:.2f}s")
        
        # 分析结果
        print("\n🔍 结果分析:")
        high_score_count = sum(1 for s in scores if s > 0.6)
        medium_score_count = sum(1 for s in scores if 0.4 <= s <= 0.6)
        low_score_count = sum(1 for s in scores if s < 0.4)
        
        print(f"  高分 (>0.6): {high_score_count} 个 ({high_score_count/len(scores)*100:.1f}%)")
        print(f"  中等 (0.4-0.6): {medium_score_count} 个 ({medium_score_count/len(scores)*100:.1f}%)")
        print(f"  低分 (<0.4): {low_score_count} 个 ({low_score_count/len(scores)*100:.1f}%)")
        
        # 保存结果
        if save_results:
            output_path = project_root / "image_mesh_evaluation_results.json"
            with open(output_path, 'w') as f:
                json.dump({
                    'results': results_sorted,
                    'statistics': {
                        'mean_score': float(np.mean(scores)),
                        'std_score': float(np.std(scores)),
                        'max_score': float(np.max(scores)),
                        'min_score': float(np.min(scores)),
                        'total_pairs': len(results),
                        'high_score_count': high_score_count,
                        'medium_score_count': medium_score_count,
                        'low_score_count': low_score_count,
                        'mean_processing_time': float(np.mean(times)),
                        'total_processing_time': float(total_time)
                    }
                }, f, indent=2)
            print(f"📄 结果已保存至: {output_path}")
        
        return True
    else:
        print("❌ 没有成功的评分结果")
        return False


def test_recall_at_k(num_test=None, save_results=False):
    """测试 Recall@K 指标"""
    print("🚀 开始测试 Recall@K 指标...")
    
    # 数据集路径
    dataset_root = project_root / "dataset" / "eval3d"
    
    # 查找配对数据
    print("🔍 正在查找 image-mesh 配对数据...")
    pairs = find_image_mesh_pairs(dataset_root)
    print(f"📁 找到 {len(pairs)} 个配对数据")
    
    if len(pairs) == 0:
        print("❌ 没有找到配对数据")
        return False
    
    # 初始化评分器
    print("🔄 正在初始化 Uni3D 评分器...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = Uni3DScorer(device=device, dtype=torch.float32)
        print("✅ Uni3D 评分器初始化成功")
    except Exception as e:
        print(f"❌ Uni3D 评分器初始化失败: {e}")
        return False
    
    # 选择测试的配对数量
    if num_test is None:
        test_pairs = pairs
    else:
        test_pairs = pairs[:num_test]
    
    # 计算 Recall@K
    recall_results, detailed_results = compute_recall_at_k(scorer, test_pairs, device)
    
    # 生成报告
    print("\n" + "="*100)
    print("🎯 Recall@K 检索准确率报告")
    print("="*100)
    
    # 显示 Recall@K 结果
    print("📊 Recall@K 结果:")
    for metric, value in recall_results.items():
        print(f"  {metric}: {value:.4f} ({value*100:.1f}%)")
    
    # 显示详细结果
    print(f"\n📋 详细匹配结果:")
    print(f"{'图像名称':<25} {'正确排名':<10} {'正确相似度':<12} {'状态'}")
    print("-" * 70)
    
    correct_matches = 0
    for result in detailed_results:
        status = "✅ 正确" if result['correct_rank'] == 1 else f"❌ 排名{result['correct_rank']}"
        if result['correct_rank'] == 1:
            correct_matches += 1
        
        print(f"{result['image_name']:<25} {result['correct_rank']:<10} {result['correct_similarity']:<12.4f} {status}")
    
    print(f"\n🎯 总结:")
    print(f"  总测试对数: {len(detailed_results)}")
    print(f"  Recall@1 (完全匹配): {correct_matches}/{len(detailed_results)} ({correct_matches/len(detailed_results)*100:.1f}%)")
    
    # 显示一些匹配失败的案例
    failed_cases = [r for r in detailed_results if r['correct_rank'] > 1]
    if failed_cases:
        print(f"\n❌ 匹配失败案例 (前5个):")
        for i, case in enumerate(failed_cases[:5]):
            print(f"  {i+1}. {case['image_name']} -> 正确排名: {case['correct_rank']}")
            print(f"     Top 3 匹配:")
            for rank, match in enumerate(case['top_5_matches'][:3], 1):
                marker = "✅" if match['mesh_name'] == case['image_name'] else "❌"
                print(f"       {rank}. {match['mesh_name']}: {match['similarity']:.4f} {marker}")
    
    # 保存结果
    if save_results:
        output_path = project_root / "recall_at_k_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                'recall_metrics': recall_results,
                'detailed_results': detailed_results,
                'summary': {
                    'total_pairs': len(detailed_results),
                    'correct_matches': correct_matches,
                    'recall_at_1': correct_matches / len(detailed_results) if detailed_results else 0
                }
            }, f, indent=2)
        print(f"📄 Recall@K 结果已保存至: {output_path}")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试 Image-Mesh 配对数据的语义一致性')
    parser.add_argument('--num_test', type=int, default=None, help='测试的配对数量 (默认测试全部)')
    parser.add_argument('--save_results', action='store_true', help='保存结果到 JSON 文件')
    parser.add_argument('--quick', action='store_true', help='快速测试（仅前10个配对）')
    parser.add_argument('--recall_test', action='store_true', help='测试 Recall@K 指标')
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_test = 10
    
    if args.recall_test:
        success = test_recall_at_k(num_test=args.num_test, save_results=args.save_results)
    else:
        success = test_image_mesh_pairs(num_test=args.num_test, save_results=args.save_results)
    
    if success:
        if args.recall_test:
            print("\n🎉 Recall@K 测试完成！")
            print("💡 该测试验证了每个图像是否与对应的 mesh 具有最高相似度")
        else:
            print("\n🎉 Image-Mesh 配对测试完成！")
            print("💡 该测试验证了 Uni3D 评分器处理 image-mesh 语义对齐的能力")
        print("📋 使用 --save_results 参数保存详细结果")
        print("⚡ 使用 --quick 参数进行快速测试")
        print("🎯 使用 --recall_test 参数测试 Recall@K 指标")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 