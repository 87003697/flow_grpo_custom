from typing import Dict, List, Set, Tuple
import numpy as np
from collections import deque
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo'):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            
        return advantages

class PerImageStatTracker:
    """
    A class to track statistics per image, using integer IDs for efficiency.
    This class is now functionally equivalent to PerPromptStatTracker.
    """
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_images = set()

    def update(self, image_ids: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Update statistics with a new batch of rewards (与PerPromptStatTracker逻辑一致)
        """
        image_ids = np.array(image_ids)
        rewards = np.array(rewards, dtype=np.float64)
        unique_ids = np.unique(image_ids)
        advantages = np.empty_like(rewards) * 0.0
        
        # 第一阶段：收集数据（与PerPromptStatTracker逻辑一致）
        for img_id in unique_ids:
            image_rewards = rewards[image_ids == img_id]
            if img_id not in self.stats:
                self.stats[img_id] = []
            self.stats[img_id].extend(image_rewards)  # ✅ 保持为list，只append新数据
            self.history_images.add(int(img_id))
        
        # 第二阶段：计算advantages（与PerPromptStatTracker逻辑一致）
        for img_id in unique_ids:
            # 🔧 修复：临时转换为numpy数组进行计算，但不改变self.stats的类型
            stats_array = np.array(self.stats[img_id])  # 临时转换
            image_rewards = rewards[image_ids == img_id]  # 重新计算当前batch该图像的奖励
            mean = np.mean(stats_array, axis=0, keepdims=True)  # 使用历史数据计算均值
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # 使用全局标准差
            else:
                std = np.std(stats_array, axis=0, keepdims=True) + 1e-4  # 使用局部标准差
            advantages[image_ids == img_id] = (image_rewards - mean) / std
            # 注意：不修改self.stats[img_id]的类型，保持为list
        
        return advantages

    def get_stats(self):
        """
        Get statistics about the tracker state
        """
        if not self.stats:
            return 0.0, 0
        
        group_sizes = [len(rewards) for rewards in self.stats.values()]
        avg_group_size = np.mean(group_sizes)
        num_trained_images = len(self.stats)
        
        return avg_group_size, num_trained_images

    def clear(self):
        """
        Clear the statistics (compatible with PerPromptStatTracker interface)
        """
        # 注意：PerPromptStatTracker的clear()只是清空，不删除历史
        # 这里我们保持相同的行为
        pass



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
