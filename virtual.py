import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)  # 支持中文路径
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def compute_colored_gradient(image):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 计算 Sobel 梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)

    # 对数增强 + 归一化
    gradient = np.log1p(gradient)
    norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    # 使用彩色伪色图
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_original_and_gradient(image_path):
    img = imread_unicode(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gradient_map = compute_colored_gradient(img_rgb)

    # 保存原图
    cv2.imwrite("../TR_Fuse/original_image.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # 保存热力图
    cv2.imwrite("../TR_Fuse/gradient_heatmap.png", cv2.cvtColor(gradient_map, cv2.COLOR_RGB2BGR))

    print("✅ 使用 OpenCV 完全无白边保存完成")

# def visualize_original_and_gradient(image_path):
#     img = cv2.imread(image_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gradient_map = compute_colored_gradient(img_rgb)
#
#     # === 保存原图 ===
#     plt.figure(figsize=(3, 3))
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     # plt.title("Original Image", fontsize=14)
#     plt.subplots_adjust(0, 0, 1, 1)  # 去除 padding
#     plt.savefig('original_image.png', dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     plt.close()
#
#     # === 保存热力图 ===
#     plt.figure(figsize=(3, 3))
#     plt.imshow(gradient_map)
#     plt.axis('off')
#     # plt.title("Color Gradient Heatmap", fontsize=14)
#     plt.subplots_adjust(0, 0, 1, 1)  # 去除 padding
#     plt.savefig('gradient_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     plt.close()
#
#     print("✅ Saved: original_image.png and gradient_heatmap.png")

# ==== 用法 ====
visualize_original_and_gradient(r"C:/Users/Administrator/Documents/图片3_5.png")  # ← 改成你的图片路径