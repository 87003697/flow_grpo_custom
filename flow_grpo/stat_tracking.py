import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        
        # 🔧 修复：收集每个prompt的奖励历史，保持数据为列表格式
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            # 🔧 修复：将numpy数组转换为列表再extend，避免类型混乱
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        
        # 🔧 修复：计算每个prompt的优势，临时转换为数组但不修改原始列表
        for prompt in unique:
            # 🔧 关键修复：临时转换为数组进行计算，但保持self.stats[prompt]为列表
            current_stats_array = np.array(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(current_stats_array, axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(current_stats_array, axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}


class PerImageStatTracker:
    """
    统计跟踪器，用于基于图像的奖励标准化
    与 PerPromptStatTracker 功能相同，但名称和注释更符合图像-3D生成场景
    """
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_images = set()

    def update(self, image_paths, rewards, exp=False):
        """
        更新图像统计信息并计算优势
        
        Args:
            image_paths: 图像路径列表
            rewards: 奖励数组
            exp: 是否使用指数奖励 (for RWR)
            
        Returns:
            advantages: 标准化的优势值
        """
        image_paths = np.array(image_paths)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(image_paths)
        advantages = np.empty_like(rewards) * 0.0
        
        # 🔧 修复：收集每个图像的奖励历史，保持数据为列表格式
        for image_path in unique:
            image_rewards = rewards[image_paths == image_path]
            if image_path not in self.stats:
                self.stats[image_path] = []
            # 🔧 修复：将numpy数组转换为列表再extend，避免类型混乱
            self.stats[image_path].extend(image_rewards.tolist())
            self.history_images.add(hash(image_path))  # 添加图像hash到历史记录
        
        # 🔧 修复：计算每个图像的优势，临时转换为数组但不修改原始列表
        for image_path in unique:
            # 🔧 关键修复：临时转换为数组进行计算，但保持self.stats[image_path]为列表
            current_stats_array = np.array(self.stats[image_path])
            image_rewards = rewards[image_paths == image_path]  # 重新计算每个图像的奖励
            mean = np.mean(current_stats_array, axis=0, keepdims=True)
            
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # 使用全局标准差
            else:
                std = np.std(current_stats_array, axis=0, keepdims=True) + 1e-4  # 使用图像特定标准差
            
            advantages[image_paths == image_path] = (image_rewards - mean) / std
        
        return advantages

    def get_stats(self):
        """获取统计信息"""
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_images = len(self.history_images)
        return avg_group_size, history_images
    
    def clear(self):
        """清除统计信息"""
        self.stats = {}


def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)

if __name__ == "__main__":
    main()
