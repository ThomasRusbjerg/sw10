import os
import json

def main():
    img_mappings = []
    for i, img in enumerate(os.listdir("data/MUSCIMA++/v2.0/data/images")):
        img_name = img[:-4]
        # print(img_name)
        img_mappings.append({"name": img_name, "id": i})
    with open('data/MUSCIMA++/v2.0/mapping_img.json', 'w') as f:
        json.dump(img_mappings, f)

if __name__ == "__main__":
    main()
