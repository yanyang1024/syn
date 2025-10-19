import cv2
import numpy as np
import logging
import os
from typing import List, Tuple, Optional
from config import Config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolygonDetector:
    """多边形检测器类"""
    
    def __init__(self, config: Config = None, debug: bool = False):
        self.config = config or Config()
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("PolygonDetector初始化完成，调试模式已启用")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        logger.info(f"正在加载图像: {image_path}")
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            error_msg = f"图像文件不存在: {image_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 检查文件扩展名
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            logger.warning(f"文件扩展名 {file_ext} 可能不被支持，支持的格式: {valid_extensions}")
        
        image = cv2.imread(image_path)
        if image is None:
            error_msg = f"无法加载图像: {image_path}，可能是文件损坏或格式不支持"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"图像加载成功，尺寸: {image.shape}")
        if self.debug:
            logger.debug(f"图像数据类型: {image.dtype}, 通道数: {image.shape[2] if len(image.shape) == 3 else 1}")
        
        return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：转换为灰度图并进行二值化"""
        logger.info("开始图像预处理")
        
        if len(image.shape) != 3:
            logger.warning(f"输入图像不是3通道彩色图像，形状: {image.shape}")
            if len(image.shape) == 2:
                gray = image
                logger.info("输入已是灰度图像，跳过灰度转换")
            else:
                raise ValueError(f"不支持的图像格式，形状: {image.shape}")
        else:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug(f"灰度转换完成，新尺寸: {gray.shape}")
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        logger.debug("高斯模糊去噪完成")
        
        # 二值化
        threshold_value = self.config.BINARY_THRESHOLD
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 统计二值化结果
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        
        logger.info(f"二值化完成，阈值: {threshold_value}")
        logger.info(f"白色像素: {white_pixels} ({white_pixels/total_pixels*100:.1f}%)")
        logger.info(f"黑色像素: {black_pixels} ({black_pixels/total_pixels*100:.1f}%)")
        
        if white_pixels < total_pixels * 0.01 or black_pixels < total_pixels * 0.01:
            logger.warning("二值化结果可能不理想，建议调整阈值")
        
        return binary
    
    def find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """查找轮廓"""
        logger.info("开始查找轮廓")
        
        if binary_image is None or binary_image.size == 0:
            logger.error("输入的二值图像为空")
            return []
        
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"找到 {len(contours)} 个轮廓")
        
        # 过滤轮廓：根据面积筛选
        filtered_contours = []
        area_stats = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_stats.append(area)
            
            if self.config.MIN_CONTOUR_AREA <= area <= self.config.MAX_CONTOUR_AREA:
                filtered_contours.append(contour)
                if self.debug:
                    logger.debug(f"轮廓 {i}: 面积 {area:.2f} - 保留")
            else:
                if self.debug:
                    logger.debug(f"轮廓 {i}: 面积 {area:.2f} - 过滤掉 (范围: {self.config.MIN_CONTOUR_AREA}-{self.config.MAX_CONTOUR_AREA})")
        
        # 统计信息
        if area_stats:
            logger.info(f"轮廓面积统计 - 最小: {min(area_stats):.2f}, 最大: {max(area_stats):.2f}, 平均: {np.mean(area_stats):.2f}")
        
        logger.info(f"过滤后保留 {len(filtered_contours)} 个轮廓")
        
        if len(filtered_contours) == 0:
            logger.warning("没有轮廓通过面积过滤，建议调整MIN_CONTOUR_AREA和MAX_CONTOUR_AREA参数")
        
        return filtered_contours
    
    def approximate_polygon(self, contour: np.ndarray) -> np.ndarray:
        """将轮廓近似为多边形"""
        epsilon = self.config.EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def get_bounding_rect(self, polygon: np.ndarray) -> Tuple[int, int, int, int]:
        """获取多边形的边界矩形"""
        x, y, w, h = cv2.boundingRect(polygon)
        return x, y, w, h
    
    def detect_polygons(self, image_path: str) -> Tuple[List[dict], np.ndarray]:
        """检测图像中的所有多边形"""
        logger.info(f"开始检测多边形: {image_path}")
        
        try:
            # 加载和预处理图像
            image = self.load_image(image_path)
            binary = self.preprocess_image(image)
            
            # 保存中间结果用于调试
            if self.debug and hasattr(self.config, 'SAVE_INTERMEDIATE_RESULTS') and self.config.SAVE_INTERMEDIATE_RESULTS:
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cv2.imwrite(os.path.join(debug_dir, f"{base_name}_binary.png"), binary)
                logger.debug(f"二值化图像已保存到调试目录")
            
            # 查找轮廓
            contours = self.find_contours(binary)
            
            if len(contours) == 0:
                logger.warning("未检测到任何符合条件的轮廓")
                return [], image
            
            # 提取多边形信息
            polygons = []
            for i, contour in enumerate(contours):
                try:
                    # 近似为多边形
                    polygon = self.approximate_polygon(contour)
                    
                    # 检查多边形是否有效
                    if len(polygon) < 3:
                        logger.warning(f"轮廓 {i} 近似后顶点数少于3，跳过")
                        continue
                    
                    # 获取边界矩形
                    x, y, w, h = self.get_bounding_rect(polygon)
                    
                    # 计算面积和周长
                    area = cv2.contourArea(polygon)
                    perimeter = cv2.arcLength(polygon, True)
                    
                    # 计算形状特征
                    aspect_ratio = w / h if h > 0 else 0
                    extent = area / (w * h) if (w * h) > 0 else 0
                    solidity = area / cv2.contourArea(cv2.convexHull(polygon)) if cv2.contourArea(cv2.convexHull(polygon)) > 0 else 0
                    
                    polygon_info = {
                        'id': i,
                        'contour': contour,
                        'polygon': polygon,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'perimeter': perimeter,
                        'vertices': len(polygon),
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'solidity': solidity
                    }
                    
                    polygons.append(polygon_info)
                    
                    if self.debug:
                        logger.debug(f"多边形 {i}: 顶点数={len(polygon)}, 面积={area:.2f}, "
                                   f"周长={perimeter:.2f}, 长宽比={aspect_ratio:.2f}")
                
                except Exception as e:
                    logger.error(f"处理轮廓 {i} 时发生错误: {str(e)}")
                    continue
            
            logger.info(f"成功检测到 {len(polygons)} 个有效多边形")
            return polygons, image
            
        except Exception as e:
            logger.error(f"检测多边形时发生错误: {str(e)}")
            raise
    
    def visualize_polygons(self, image: np.ndarray, polygons: List[dict], 
                          unique_polygon_id: int = None) -> np.ndarray:
        """可视化多边形"""
        result_image = image.copy()
        
        for polygon_info in polygons:
            color = self.config.UNIQUE_COLOR if polygon_info['id'] == unique_polygon_id else self.config.CONTOUR_COLOR
            
            # 绘制多边形
            cv2.drawContours(result_image, [polygon_info['polygon']], -1, color, self.config.LINE_THICKNESS)
            
            # 绘制边界矩形
            x, y, w, h = polygon_info['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 1)
            
            # 添加ID标签
            cv2.putText(result_image, str(polygon_info['id']), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image