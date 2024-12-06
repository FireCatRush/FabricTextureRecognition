import numpy as np
import cv2
from skimage import feature
from scipy import ndimage
from sklearn.cluster import DBSCAN
from scipy.spatial import distance


class FabricTextureAnalyzer:
    def __init__(self):
        self.directions = np.array([0, 45, 90, 135])  # 纹理主要方向

    def preprocess_image(self, image):
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 双边滤波保持边缘
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return denoised

    def extract_gabor_features(self, image):
        """使用Gabor滤波器提取纹理特征"""
        features = []
        for theta in self.directions:
            # 创建不同方向的Gabor核
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta * np.pi / 180, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            features.append(filtered)

        return np.array(features)

    def calculate_lbp(self, image):
        """计算局部二值模式(LBP)特征"""
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        return lbp

    def detect_edges(self, image):
        """边缘检测增强"""
        # Canny边缘检测
        edges = cv2.Canny(image, 50, 150)

        # 形态学操作增强连续性
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def extract_texture_direction(self, edges):
        """提取纹理方向"""
        # Hough变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return None

        # 统计主要方向
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)

        # 使用DBSCAN聚类找到主要方向
        angles = np.array(angles).reshape(-1, 1)
        clustering = DBSCAN(eps=5, min_samples=3).fit(angles)

        if len(set(clustering.labels_)) <= 1:
            return None

        # 返回最大聚类的中心角度
        largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
        main_angles = angles[clustering.labels_ == largest_cluster]
        return np.median(main_angles)

    def segment_texture_regions(self, image, gabor_features, lbp):
        """分割纹理区域"""
        # 组合Gabor和LBP特征
        combined_features = np.concatenate([
            gabor_features.reshape(len(self.directions), -1),
            lbp.reshape(1, -1)
        ], axis=0)

        # 使用K-means聚类
        features_normalized = combined_features / np.max(combined_features)
        features_reshaped = features_normalized.reshape(-1, image.shape[0] * image.shape[1]).T

        # DBSCAN聚类
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(features_reshaped)

        # 重建分割掩码
        mask = clustering.labels_.reshape(image.shape)
        return mask

    def analyze_texture(self, image_path):
        """主要分析流程"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像")

        # 预处理
        preprocessed = self.preprocess_image(image)

        # 提取特征
        gabor_features = self.extract_gabor_features(preprocessed)
        lbp_features = self.calculate_lbp(preprocessed)

        # 边缘检测
        edges = self.detect_edges(preprocessed)

        # 提取方向
        main_direction = self.extract_texture_direction(edges)

        # 分割纹理区域
        texture_mask = self.segment_texture_regions(preprocessed, gabor_features, lbp_features)

        return {
            'original': image,
            'preprocessed': preprocessed,
            'edges': edges,
            'texture_mask': texture_mask,
            'main_direction': main_direction
        }

    def visualize_results(self, results):
        """可视化分析结果"""
        # 创建输出图像
        visualization = np.zeros_like(results['original'])

        # 为不同纹理区域着色
        unique_labels = np.unique(results['texture_mask'])
        colors = np.random.randint(0, 255, (len(unique_labels), 3))

        for label in unique_labels:
            if label == -1:  # DBSCAN噪声点
                continue
            mask = results['texture_mask'] == label
            visualization[mask] = colors[label]

        # 叠加原始图像
        output = cv2.addWeighted(results['original'], 0.7, visualization, 0.3, 0)

        # 如果存在主方向，绘制方向线
        if results['main_direction'] is not None:
            h, w = output.shape[:2]
            center = (w // 2, h // 2)
            length = 100
            angle = results['main_direction']
            end_point = (
                int(center[0] + length * np.cos(angle * np.pi / 180)),
                int(center[1] + length * np.sin(angle * np.pi / 180))
            )
            cv2.line(output, center, end_point, (0, 255, 0), 2)

        return output


def demo_texture_analysis(image_path):
    """演示使用方法"""
    analyzer = FabricTextureAnalyzer()

    try:
        # 分析纹理
        results = analyzer.analyze_texture(image_path)

        # 可视化结果
        output = analyzer.visualize_results(results)

        # 显示结果
        cv2.imshow('Texture Analysis Results', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return results

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return None


# 使用示例
image_path = "data/1.png"
results = demo_texture_analysis(image_path)

# 如果需要保存结果
if results is not None:
    cv2.imwrite("texture_analysis_result.jpg", results['texture_mask'])