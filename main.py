import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def preprocess_image(image_path):
    """
    Preprocess the input image for texture line detection.

    Args:
        image_path (str): Path to the input image file

    Returns:
        tuple: (original_image, binary_image)
            - original_image: Grayscale image array
            - binary_image: Binary image array after preprocessing

    Processing steps:
        1. Load and convert image to grayscale
        2. Apply LoG filter for texture enhancement
        3. Perform morphological closing
        4. Apply binary thresholding
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Extract texture information using LoG filter
    log_image = cv2.Laplacian(image, cv2.CV_64F)
    log_image = cv2.GaussianBlur(log_image, (5, 5), 0)
    log_image = cv2.convertScaleAbs(log_image)  # Convert negative values to positive range

    # Remove small dark regions using morphological closing
    kernel = np.ones((10, 10), np.uint8)
    closed_image = cv2.morphologyEx(log_image, cv2.MORPH_CLOSE, kernel)

    # Create binary image
    _, binary_image = cv2.threshold(closed_image, 10, 255, cv2.THRESH_BINARY)
    return image, binary_image


def calculate_adaptive_change(binary_image, current_y, current_x, window_size=15):
    """
    Calculate adaptive search window size based on local texture complexity.

    Args:
        binary_image (np.ndarray): Binary image array
        current_y (int): Current Y coordinate
        current_x (int): Current X coordinate
        window_size (int, optional): Size of local window. Defaults to 15

    Returns:
        int: Maximum allowed change in pixel position based on texture complexity

    Algorithm:
        1. Extract local window around current position
        2. Calculate texture complexity using standard deviation
        3. Adjust change amount based on complexity
    """
    height, width = binary_image.shape
    window = binary_image[max(0, current_y - window_size):min(height, current_y + window_size + 1),
             max(0, current_x - window_size):min(width, current_x + window_size + 1)]

    # Calculate texture complexity using standard deviation
    texture_complexity = np.std(window)
    base_ratio = 0.1
    complexity_factor = min(texture_complexity / 128, 1.0)

    # Calculate maximum position change based on complexity
    max_change = int(window_size * base_ratio * (1 + complexity_factor))
    return max_change


def find_start_points(binary_image, min_distance=30):
    """
    Find suitable starting points for texture line tracing.

    Args:
        binary_image (np.ndarray): Binary image array
        min_distance (int, optional): Minimum distance between start points. Defaults to 30

    Returns:
        list: List of tuples containing (x, y) coordinates of starting points

    Algorithm:
        1. Scan image vertically with 10-pixel intervals
        2. Look for white pixels (255) in unvisited areas
        3. Validate potential start points using local window analysis
        4. Ensure minimum distance between selected points
    """
    start_points = []
    visited = np.zeros_like(binary_image)
    height, width = binary_image.shape

    for y in range(0, height, 10):
        for x in range(width):
            if binary_image[y, x] == 255 and visited[y, x] == 0:
                window = binary_image[max(0, y - 5):min(height, y + 6),
                         max(0, x - 5):min(width, x + 6)]
                if np.sum(window == 255) > 20:
                    if not start_points or all(abs(y - py) > min_distance for px, py in start_points):
                        start_points.append((x, y))
                        visited[max(0, y - min_distance):min(height, y + min_distance + 1),
                        max(0, x - min_distance):min(width, x + min_distance + 1)] = 1
    return start_points


def trace_texture_line(binary_image, start_point, min_length=20):
    """
    Trace a texture line from a given starting point.

    Args:
        binary_image (np.ndarray): Binary image array
        start_point (tuple): (x, y) coordinates of starting point
        min_length (int, optional): Minimum required length of line. Defaults to 20

    Returns:
        list or None: List of (x, y) coordinates forming the line if length >= min_length,
                     None otherwise

    Algorithm:
        1. Trace line rightward and leftward from start point
        2. Use adaptive window to follow texture pattern
        3. Track visited pixels to avoid loops
        4. Apply adaptive position changes based on local texture
    """
    x, y = start_point
    height, width = binary_image.shape
    line_points = [(x, y)]
    visited = np.zeros_like(binary_image)
    window_size = 50

    # Right-direction tracing
    current_x, current_y = x, y
    while current_x < width - window_size:
        search_window = binary_image[max(0, current_y - window_size):min(height, current_y + window_size + 1),
                        current_x:min(width, current_x + window_size)]
        white_pixels = np.where(search_window == 255)

        if len(white_pixels[0]) == 0:
            break

        next_y = current_y - window_size + int(np.mean(white_pixels[0]))
        next_x = current_x + int(np.mean(white_pixels[1]))

        max_y_change = calculate_adaptive_change(binary_image, current_y, current_x, window_size)
        next_y = np.clip(next_y, current_y - max_y_change, current_y + max_y_change)

        if visited[next_y, next_x] == 1:
            break

        line_points.append((next_x, next_y))
        visited[next_y, next_x] = 1
        current_x, current_y = next_x, next_y

    # Similar process for left-direction tracing
    current_x, current_y = x, y
    left_points = []
    while current_x > window_size:
        search_window = binary_image[max(0, current_y - window_size):min(height, current_y + window_size + 1),
                        max(0, current_x - window_size):current_x]
        white_pixels = np.where(search_window == 255)

        if len(white_pixels[0]) == 0:
            break

        next_y = current_y - window_size + int(np.mean(white_pixels[0]))
        next_x = current_x - window_size + int(np.mean(white_pixels[1]))

        max_y_change = calculate_adaptive_change(binary_image, current_y, current_x, window_size)
        next_y = np.clip(next_y, current_y - max_y_change, current_y + max_y_change)

        if visited[next_y, next_x] == 1:
            break

        left_points.append((next_x, next_y))
        visited[next_y, next_x] = 1
        current_x, current_y = next_x, next_y

    all_points = list(reversed(left_points)) + line_points
    return all_points if len(all_points) >= min_length else None


def fit_multiple_textures(binary_image):
    """
    Detect and trace multiple texture lines in the image.

    Args:
        binary_image (np.ndarray): Binary image array

    Returns:
        list: List of valid texture lines, where each line is a list of (x, y) coordinates

    Process:
        1. Find suitable starting points
        2. Trace texture line from each starting point
        3. Filter out invalid/short lines
    """
    start_points = find_start_points(binary_image)
    all_lines = []
    for start_point in start_points:
        line = trace_texture_line(binary_image, start_point)
        if line is not None:
            all_lines.append(line)
    return all_lines


def visualize_results(image, binary_image, all_lines):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(133)
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i, line in enumerate(all_lines):
        color = colors[i % len(colors)]
        for j in range(len(line) - 1):
            pt1 = tuple(map(int, line[j]))
            pt2 = tuple(map(int, line[j + 1]))
            cv2.line(result, pt1, pt2, color, 2)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f'Fitted Results ({len(all_lines)} lines)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def select_optimal_line(all_lines, binary_image):
    height, width = binary_image.shape
    lines_features = [analyze_line_features(line, binary_image) for line in all_lines]

    # 归一化特征
    max_length = max(f['length'] for f in lines_features)
    max_centrality = max(f['centrality'] for f in lines_features)

    scores = []
    for features in lines_features:
        length_score = features['length'] / max_length
        curvature_score = 1 - (features['max_curvature'] / np.pi)  # 曲率越小越好
        centrality_score = 1 - (features['centrality'] / max_centrality)  # 中心度越高越好

        # 综合评分
        total_score = (0.4 * length_score +
                       0.3 * curvature_score +
                       0.3 * centrality_score)
        scores.append(total_score)

    best_line_idx = np.argmax(scores)
    return all_lines[best_line_idx], scores[best_line_idx]


def analyze_line_features(line_points, binary_image):
    """
    Calculate geometric features of a traced line.

    Args:
        line_points (list): List of (x, y) coordinates forming the line
        binary_image (np.ndarray): Binary image array

    Returns:
        dict: Dictionary containing line features:
            - length: Number of points in the line
            - max_curvature: Maximum angle between consecutive segments
            - avg_curvature: Average angle between consecutive segments
            - centrality: Average distance from image center

    Features:
        - Curvature calculated using angle between consecutive line segments
        - Centrality measured as average distance from image center
    """
    if not line_points:
        return {
            'length': 0,
            'max_curvature': 0,
            'avg_curvature': 0,
            'centrality': float('inf')
        }

    points = np.array(line_points)
    height, width = binary_image.shape
    length = len(points)

    # Calculate segment angles and curvature
    curvatures = []
    if length > 2:
        for i in range(1, length - 1):
            p1, p2, p3 = points[i - 1:i + 2]
            v1 = p1 - p2
            v2 = p3 - p2
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            curvatures.append(angle)

    max_curvature = max(curvatures) if curvatures else 0
    avg_curvature = np.mean(curvatures) if curvatures else 0

    # Calculate average distance from image center
    image_center = np.array([width // 2, height // 2])
    centrality = np.mean([np.linalg.norm(np.array([x, y]) - image_center) for x, y in points])

    return {
        'length': length,
        'max_curvature': max_curvature,
        'avg_curvature': avg_curvature,
        'centrality': centrality
    }


def visualize_results_with_best(image, binary_image, all_lines, best_line):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')

    plt.subplot(133)
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw all lines in light colors
    for i, line in enumerate(all_lines):
        for j in range(len(line) - 1):
            pt1 = tuple(map(int, line[j]))
            pt2 = tuple(map(int, line[j + 1]))
            cv2.line(result, pt1, pt2, (100, 100, 100), 1)

    # Draw best line in green
    for j in range(len(best_line) - 1):
        pt1 = tuple(map(int, best_line[j]))
        pt2 = tuple(map(int, best_line[j + 1]))
        cv2.line(result, pt1, pt2, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Fitted Results (Best Line in Green)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def create_curve_spectrogram(image, curve_points, window_size=50):
    curve = np.array(curve_points)
    x, y = curve[:, 0], curve[:, 1]

    # 移除重复的x坐标
    unique_indices = np.unique(x, return_index=True)[1]
    x = x[unique_indices]
    y = y[unique_indices]

    # 确保x坐标严格递增
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = f(x_new)

    height, width = image.shape
    upper_spectrogram = []
    lower_spectrogram = []

    for i in range(len(x_new) - 1):
        current_x = int(x_new[i])
        current_y = int(y_new[i])

        upper_slice = image[max(0, current_y - window_size):current_y, current_x]
        lower_slice = image[current_y:min(height, current_y + window_size), current_x]

        if len(upper_slice) > 0:
            upper_spectrogram.append(np.mean(upper_slice))
        if len(lower_slice) > 0:
            lower_spectrogram.append(np.mean(lower_slice))

    return np.array(upper_spectrogram), np.array(lower_spectrogram), x_new


def visualize_spectrograms(image, binary_image, curve_points, upper_spec, lower_spec, x_coords):
    plt.figure(figsize=(15, 10))

    # 原图显示
    plt.subplot(311)
    result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    curve = np.array(curve_points)

    for x, y in curve:
        cv2.line(result, (int(x), int(y) - 10), (int(x), int(y) + 10), (0, 255, 0), 1)

    plt.imshow(result)
    plt.plot(curve[:, 0], curve[:, 1], 'r-', linewidth=2)
    plt.title('Local Sampling Region')
    plt.axis('off')

    # 上方波形图
    plt.subplot(312)
    plt.plot(x_coords[:-1], upper_spec, 'b-', linewidth=1)
    # 添加水平参考线表示曲线位置
    plt.axhline(y=np.mean(upper_spec), color='r', linestyle='--', label='Curve Position')
    plt.title('Upper Region Intensity Profile')
    plt.grid(True)
    plt.ylabel('Intensity')
    plt.legend()

    # 下方波形图
    plt.subplot(313)
    plt.plot(x_coords[:-1], lower_spec, 'r-', linewidth=1)
    plt.axhline(y=np.mean(lower_spec), color='r', linestyle='--', label='Curve Position')
    plt.title('Lower Region Intensity Profile')
    plt.grid(True)
    plt.xlabel('Position (pixels)')
    plt.ylabel('Intensity')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(image_path):
    original_image, binary_image = preprocess_image(image_path)
    all_lines = fit_multiple_textures(binary_image)
    best_line, score = select_optimal_line(all_lines, binary_image)
    visualize_results_with_best(original_image, binary_image, all_lines, best_line)
    upper_spec, lower_spec, x_coords = create_curve_spectrogram(original_image, best_line)
    visualize_spectrograms(original_image, binary_image, best_line, upper_spec, lower_spec, x_coords)


if __name__ == "__main__":
    image_path = 'data/2.png'
    main(image_path)