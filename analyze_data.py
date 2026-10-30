import os
from PIL import Image
import json

dataset_path = r"e:\AIML_report"
splits = ["train", "val", "test", "auto_test"]

stats = {
    "total_images": 0,
    "splits": {},
    "classes": set(),
    "image_sizes": set(),
    "image_formats": set()
}

for split in splits:
    split_path = os.path.join(dataset_path, split)
    if not os.path.exists(split_path):
        continue
    
    stats["splits"][split] = {}
    
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
            
        stats["classes"].add(cls)
        stats["splits"][split][cls] = 0
        
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            if not os.path.isfile(img_path):
                continue
                
            stats["splits"][split][cls] += 1
            stats["total_images"] += 1
            
            # Check 5 images per class per split to infer dimensions and formats
            if stats["splits"][split][cls] <= 5:
                try:
                    with Image.open(img_path) as img:
                        stats["image_sizes"].add(img.size)
                        stats["image_formats"].add(img.format)
                except Exception as e:
                    pass

stats["classes"] = list(stats["classes"])
stats["image_sizes"] = [f"{w}x{h}" for w, h in stats["image_sizes"]]
stats["image_formats"] = list(stats["image_formats"])

print(json.dumps(stats, indent=4))
