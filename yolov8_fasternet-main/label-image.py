import os
import cv2

# 输入文件夹
image_folder = '/mnt/vip/lj/mydata/UTDAC2020/test/images'  # 图像文件夹
label_folder = '/mnt/vip/lj/mydata/UTDAC2020/test/labels'  # 标签文件夹

# 输出文件夹
output_folder = '/mnt/vip/hc/yolov8uw/result/8s'  # 标记后的图像输出文件夹

# 类别名称和颜色映射
labels = ["echinus",  "starfish","holothurian", "scallop"]  # 类别名称
colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
            (0, 255, 128), (128, 0, 255), (255, 128, 0)]  # BGR颜色


def draw_bboxes(image, label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        color = colormap[class_id]  # 获取对应类别的颜色

        # 边界框坐标
        x_min = int((x - w / 2) * image.shape[1])
        y_min = int((y - h / 2) * image.shape[0])
        x_max = int((x + w / 2) * image.shape[1])
        y_max = int((y + h / 2) * image.shape[0])

        # 在图像上绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return image


# 遍历图像文件夹和标签文件夹
for img_filename in os.listdir(image_folder):
    if img_filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, img_filename)
        label_filename = img_filename[:-4] + '.txt'
        label_path = os.path.join(label_folder, label_filename)

        if os.path.exists(label_path):
            image = cv2.imread(img_path)
            image_with_bboxes = draw_bboxes(image, label_path)

            output_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(output_path, image_with_bboxes)

print("标记完成。")
