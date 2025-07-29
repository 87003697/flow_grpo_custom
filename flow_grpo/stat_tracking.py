from typing import Dict, List, Set, Tuple
import numpy as np
from collections import deque


class PerPromptStatTracker:
    """
    A class to track statistics per prompt.
    """

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for p in unique:
            prompt_rewards = rewards[prompts == p]
            if p not in self.stats:
                self.stats[p] = deque(maxlen=self.buffer_size)
            self.stats[p].extend(prompt_rewards)
            if len(self.stats[p]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[p])
                std = np.std(self.stats[p]) + 1e-6
            advantages[prompts == p] = (prompt_rewards - mean) / std
        return advantages
    
    def get_stats(self):
        """获取统计信息"""
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        trained_prompt_num = len(self.stats)
        return avg_group_size, trained_prompt_num

class PerImageStatTracker:
    """
    A class to track statistics per image, using integer IDs for efficiency.
    This class is now functionally equivalent to PerPromptStatTracker.
    """
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        # self.stats is a dictionary where keys are integer image IDs 
        # and values are deques of rewards.
        self.stats = {}

    def update(self, image_ids: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Update statistics with a new batch of rewards (与PerPromptStatTracker完全一致)
        """
        unique_ids = np.unique(image_ids)
        advantages = np.empty_like(rewards)
        
        for img_id in unique_ids:
            image_rewards = rewards[image_ids == img_id]
            if img_id not in self.stats:
                self.stats[img_id] = deque(maxlen=self.buffer_size)
            self.stats[img_id].extend(image_rewards)
            
            if len(self.stats[img_id]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[img_id])
                std = np.std(self.stats[img_id]) + 1e-6
            
            # 🔧 FIX: Prevent NaN by checking if std is zero.
            if std > 1e-6:
                advantages[image_ids == img_id] = (image_rewards - mean) / std
            else:
                advantages[image_ids == img_id] = 0.0
        
        return advantages

    def get_stats(self):
        """获取统计信息"""
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats)
        trained_image_num = len(self.stats)
        return avg_group_size, trained_image_num



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
