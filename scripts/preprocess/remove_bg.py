import os
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from tqdm import tqdm

class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.cuda()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")  # [1, 3, 1024, 1024]
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()  # [1, 1, 1024, 1024]
        pred = preds[0].squeeze()  # [1024, 1024]
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image = image.convert("RGBA")
        image.putalpha(mask)
        return image


def remove_background_batch(input_dir: str, output_dir: str):
    """
    批量去除背景并保存为 RGBA PNG
    
    Args:
        input_dir: 输入图片文件夹路径
        output_dir: 输出图片文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # 获取所有图片文件
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 初始化模型
    print("加载 BiRefNet 模型...")
    model = BiRefNet()
    
    # 处理每张图片
    for filename in tqdm(image_files, desc="处理图片"):
        input_path = os.path.join(input_dir, filename)
        
        # 输出文件名（统一为 .png）
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # 读取图片
            image = Image.open(input_path).convert("RGB")
            
            # 去除背景
            result = model(image)
            
            # 保存为 RGBA PNG
            result.save(output_path, "PNG")
        except Exception as e:
            print(f"处理 {filename} 失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量去除图片背景")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出图片文件夹")
    args = parser.parse_args()
    
    remove_background_batch(args.input_dir, args.output_dir)