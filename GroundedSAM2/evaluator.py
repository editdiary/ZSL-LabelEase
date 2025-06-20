import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import os
import glob
from collections import defaultdict
from typing import List, Dict, Any
import pandas as pd

# RLE 디코딩을 위한 간단한 구현
def decode_rle(rle_data):
    """RLE 인코딩된 마스크를 디코딩"""
    if isinstance(rle_data, dict):
        counts = rle_data['counts']
        size = rle_data['size']
    else:
        return None
    
    # 간단한 RLE 디코딩 (실제로는 pycocotools 사용 권장)
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    idx = 0
    val = 0
    
    for count in counts:
        if isinstance(count, str):
            # 문자열 형태의 counts 처리
            import re
            numbers = re.findall(r'\d+', count)
            for i, num in enumerate(numbers):
                num = int(num)
                if i % 2 == 0:  # 짝수 인덱스는 값
                    val = num
                else:  # 홀수 인덱스는 반복 횟수
                    mask[idx:idx+num] = val
                    idx += num
        else:
            # 숫자 형태의 counts 처리
            mask[idx:idx+count] = val
            idx += count
            val = 1 - val
    
    return mask.reshape(size[0], size[1])

class CompleteGroundedSAM2Evaluator:
    def __init__(self, coco_annotations_path: str, grounded_sam2_results_dir: str):
        """
        바운딩 박스와 세그멘테이션 마스크를 모두 사용하는 GroundedSAM2 모델 성능 평가기
        
        Args:
            coco_annotations_path: COCO 포맷 실제 라벨링 파일 경로
            grounded_sam2_results_dir: GroundedSAM2 추론 결과 파일들이 있는 디렉토리
        """
        self.coco_annotations_path = coco_annotations_path
        self.grounded_sam2_results_dir = grounded_sam2_results_dir
        
        # 데이터 로드
        self.coco_data = self._load_coco_annotations()
        self.result_files = self._get_result_files()
        
    def _load_coco_annotations(self) -> Dict:
        """COCO 포맷 annotation 파일 로드"""
        with open(self.coco_annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지별 annotation 그룹화
        image_annotations = defaultdict(list)
        for ann in data['annotations']:
            image_annotations[ann['image_id']].append(ann)
        
        return {
            'images': {img['id']: img for img in data['images']},
            'annotations': dict(image_annotations),
            'categories': {cat['id']: cat for cat in data['categories']}
        }
    
    def _get_result_files(self) -> List[str]:
        """결과 파일 목록 가져오기"""
        pattern = os.path.join(self.grounded_sam2_results_dir, "*_grounded_sam2_results.json")
        files = glob.glob(pattern)
        return sorted(files)
    
    def _load_grounded_sam2_results(self, result_file: str) -> Dict:
        """단일 GroundedSAM2 결과 파일 로드"""
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 파일명에서 이미지 ID 추출
        image_filename = os.path.basename(data['image_path'])
        image_id = None
        
        for img_id, img_info in self.coco_data['images'].items():
            if img_info['file_name'] == image_filename:
                image_id = img_id
                break
        
        if image_id is None:
            raise ValueError(f"이미지 {image_filename}을 COCO 데이터에서 찾을 수 없습니다.")
        
        return {
            'image_id': image_id,
            'image_path': data['image_path'],
            'annotations': data['annotations']
        }
    
    def _polygon_to_mask(self, polygon: List[float], height: int, width: int) -> np.ndarray:
        """폴리곤을 마스크로 변환"""
        polygon = np.array(polygon).reshape(-1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask
    
    def _decode_prediction_mask(self, segmentation_data: Dict) -> np.ndarray:
        """예측 결과의 마스크 디코딩"""
        return maskUtils.decode(segmentation_data).astype(np.uint8)
    
    def _calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """두 마스크 간의 IoU 계산"""
        if mask1 is None or mask2 is None:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_pixel_level_iou(self, gt_masks: List, pred_masks: List, height: int, width: int) -> float:
        """
        전체 이미지에 대한 픽셀 레벨 IoU 계산 (Foreground vs Background)
        - 모든 GT 마스크를 하나로 합침
        - 모든 예측 마스크를 하나로 합침
        - 두 통합 마스크 간의 IoU를 계산하여, 매칭과 관계없는 전체 성능을 측정
        """
        gt_foreground_mask = np.zeros((height, width), dtype=bool)
        for gt_info in gt_masks:
            if gt_info['mask'] is not None:
                gt_foreground_mask = np.logical_or(gt_foreground_mask, gt_info['mask'])

        pred_foreground_mask = np.zeros((height, width), dtype=bool)
        for pred_info in pred_masks:
            if pred_info['mask'] is not None:
                pred_foreground_mask = np.logical_or(pred_foreground_mask, pred_info['mask'])
        
        # 합쳐진 마스크 간의 IoU 계산 (0으로 나누는 경우 방지 포함)
        intersection = np.logical_and(gt_foreground_mask, pred_foreground_mask).sum()
        union = np.logical_or(gt_foreground_mask, pred_foreground_mask).sum()
        
        if union == 0:
            # GT와 예측 모두 비어있는 경우, 완벽하게 일치하므로 1.0 반환
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """두 바운딩 박스 간의 IoU 계산"""
        # COCO 포맷: [x, y, width, height]
        # GroundedSAM2 포맷: [x1, y1, x2, y2]
        
        # COCO bbox를 [x1, y1, x2, y2] 형태로 변환
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        # GroundedSAM2 bbox는 이미 [x1, y1, x2, y2] 형태
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 계산
        area1 = w1 * h1
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union
    
    def evaluate_single_image(self, result_file: str) -> Dict[str, Any]:
        """단일 이미지에 대한 평가 수행 (바운딩 박스 + 마스크)"""
        grounded_sam2_data = self._load_grounded_sam2_results(result_file)
        image_id = grounded_sam2_data['image_id']
        image_info = self.coco_data['images'][image_id]
        
        # 실제 annotation (ground truth)
        gt_annotations = self.coco_data['annotations'].get(image_id, [])
        
        # 모델 예측 결과
        pred_annotations = grounded_sam2_data['annotations']
        
        # 결과 저장
        results = {
            'image_id': image_id,
            'image_filename': image_info['file_name'],
            'gt_count': len(gt_annotations),
            'pred_count': len(pred_annotations),
            'matches': [],
            'unmatched_gt': [],
            'unmatched_pred': [],
            'metrics': {
                'bbox': {},
                'mask': {},
                'combined': {},
                'pixel_level_iou': 0.0
            }
        }
        
        # GT annotation을 마스크로 변환
        gt_masks = []
        for gt_ann in gt_annotations:
            gt_mask = None
            if 'segmentation' in gt_ann:
                if isinstance(gt_ann['segmentation'], list):
                    # 폴리곤 형태
                    gt_mask = self._polygon_to_mask(
                        gt_ann['segmentation'][0], 
                        image_info['height'], 
                        image_info['width']
                    )
                else:
                    # RLE 형태
                    gt_mask = maskUtils.decode(gt_ann['segmentation'])
            
            gt_masks.append({
                'mask': gt_mask,
                'bbox': gt_ann['bbox'],
                'category_id': gt_ann['category_id'],
                'annotation': gt_ann
            })
        
        # 예측 결과를 마스크로 변환
        pred_masks = []
        for pred_ann in pred_annotations:
            pred_mask = None
            if 'segmentation' in pred_ann:
                pred_mask = self._decode_prediction_mask(pred_ann['segmentation'])
            
            # score가 리스트가 아닌 float일 경우에 대한 예외 처리
            score_value = pred_ann['score']
            if isinstance(score_value, list) and score_value:
                score = score_value[0]
            else:
                score = score_value

            pred_masks.append({
                'mask': pred_mask,
                'bbox': pred_ann['bbox'],
                'class_name': pred_ann['class_name'],
                'score': score,
                'annotation': pred_ann
            })
        
        # 픽셀 레벨 IoU 계산
        pixel_iou = self._calculate_pixel_level_iou(
            gt_masks, pred_masks, image_info['height'], image_info['width']
        )
        results['metrics']['pixel_level_iou'] = pixel_iou

        # 바운딩 박스 기반 매칭
        bbox_matches = self._match_objects(gt_masks, pred_masks, 'bbox')
        
        # 마스크 기반 매칭
        mask_matches = self._match_objects(gt_masks, pred_masks, 'mask')
        
        # 결합 매칭 (바운딩 박스 + 마스크)
        combined_matches = self._match_objects(gt_masks, pred_masks, 'combined')
        
        # 각 방식별 메트릭 계산
        results['metrics']['bbox'] = self._calculate_metrics(bbox_matches, len(gt_masks), len(pred_masks))
        results['metrics']['mask'] = self._calculate_metrics(mask_matches, len(gt_masks), len(pred_masks))
        results['metrics']['combined'] = self._calculate_metrics(combined_matches, len(gt_masks), len(pred_masks))
        
        # 상세 매칭 정보 저장
        results['matches'] = {
            'bbox': bbox_matches,
            'mask': mask_matches,
            'combined': combined_matches
        }
        
        return results
    
    def _match_objects(self, gt_masks: List, pred_masks: List, match_type: str) -> List[Dict]:
        """객체 매칭 수행"""
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_mask_info in enumerate(pred_masks):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_mask_info in enumerate(gt_masks):
                if j in matched_gt:
                    continue
                
                if match_type == 'bbox':
                    iou = self._calculate_bbox_iou(gt_mask_info['bbox'], pred_mask_info['bbox'])
                elif match_type == 'mask':
                    iou = self._calculate_mask_iou(gt_mask_info['mask'], pred_mask_info['mask'])
                else:  # combined
                    bbox_iou = self._calculate_bbox_iou(gt_mask_info['bbox'], pred_mask_info['bbox'])
                    mask_iou = self._calculate_mask_iou(gt_mask_info['mask'], pred_mask_info['mask'])
                    iou = 0.3 * bbox_iou + 0.7 * mask_iou  # 마스크에 더 높은 가중치
                
                if iou > best_iou and iou >= 0.5:  # IoU 임계값
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
                
                matches.append({
                    'gt_idx': best_gt_idx,
                    'pred_idx': i,
                    'iou': best_iou,
                    'gt_annotation': gt_masks[best_gt_idx]['annotation'],
                    'pred_annotation': pred_masks[i]['annotation']
                })
        
        return matches
    
    def _calculate_metrics(self, matches: List[Dict], gt_count: int, pred_count: int) -> Dict:
        """메트릭 계산"""
        tp = len(matches)
        fp = pred_count - tp
        fn = gt_count - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def evaluate_all_images(self) -> Dict[str, Any]:
        """모든 이미지에 대한 평가 수행"""
        all_results = []
        summary_stats = {
            'total_images': len(self.result_files),
            'bbox': {'iou_scores': [], 'precision_scores': [], 'recall_scores': [], 'f1_scores': []},
            'mask': {'iou_scores': [], 'precision_scores': [], 'recall_scores': [], 'f1_scores': []},
            'combined': {'iou_scores': [], 'precision_scores': [], 'recall_scores': [], 'f1_scores': []},
            'pixel_level_iou_scores': []
        }
        
        print(f"총 {len(self.result_files)}개 이미지 평가 시작...")
        print("=" * 80)
        
        for i, result_file in enumerate(self.result_files, 1):
            try:
                print(f"평가 중: {i}/{len(self.result_files)} - {os.path.basename(result_file)}")
                
                result = self.evaluate_single_image(result_file)
                all_results.append(result)
                
                # 각 방식별 통계 수집
                for match_type in ['bbox', 'mask', 'combined']:
                    matches = result['matches'][match_type]
                    metrics = result['metrics'][match_type]
                    
                    # IoU 점수들 수집
                    for match in matches:
                        summary_stats[match_type]['iou_scores'].append(match['iou'])
                    
                    # 메트릭 점수들 수집
                    summary_stats[match_type]['precision_scores'].append(metrics['precision'])
                    summary_stats[match_type]['recall_scores'].append(metrics['recall'])
                    summary_stats[match_type]['f1_scores'].append(metrics['f1_score'])
                
                # 픽셀 레벨 IoU 점수 수집
                summary_stats['pixel_level_iou_scores'].append(result['metrics']['pixel_level_iou'])
                
                print(f"  - GT: {result['gt_count']}, Pred: {result['pred_count']}")
                print(f"  - BBox: P={result['metrics']['bbox']['precision']:.4f}, R={result['metrics']['bbox']['recall']:.4f}, F1={result['metrics']['bbox']['f1_score']:.4f}")
                print(f"  - Mask: P={result['metrics']['mask']['precision']:.4f}, R={result['metrics']['mask']['recall']:.4f}, F1={result['metrics']['mask']['f1_score']:.4f}")
                print(f"  - Combined: P={result['metrics']['combined']['precision']:.4f}, R={result['metrics']['combined']['recall']:.4f}, F1={result['metrics']['combined']['f1_score']:.4f}")
                print(f"  - Pixel-level IoU: {result['metrics']['pixel_level_iou']:.4f}")
                
            except Exception as e:
                print(f"  오류 발생: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return {
            'individual_results': all_results,
            'summary_stats': summary_stats
        }
    
    def print_summary_results(self, evaluation_results: Dict[str, Any]):
        """요약 결과 출력"""
        summary = evaluation_results['summary_stats']
        
        print("\n" + "=" * 100)
        print("전체 평가 결과 요약 (바운딩 박스 + 마스크 + 결합)")
        print("=" * 100)
        
        print(f"평가된 이미지 수: {summary['total_images']}")
        print()
        
        # 각 방식별 결과 출력
        for match_type in ['bbox', 'mask', 'combined']:
            print(f"{match_type.upper()} 기반 평가:")
            print(f"  평균 Precision: {np.mean(summary[match_type]['precision_scores']):.4f}")
            print(f"  평균 Recall: {np.mean(summary[match_type]['recall_scores']):.4f}")
            print(f"  평균 F1-Score: {np.mean(summary[match_type]['f1_scores']):.4f}")
            
            if summary[match_type]['iou_scores']:
                print(f"  평균 IoU: {np.mean(summary[match_type]['iou_scores']):.4f}")
                print(f"  IoU 범위: {np.min(summary[match_type]['iou_scores']):.4f} ~ {np.max(summary[match_type]['iou_scores']):.4f}")
            print()
        
        # 픽셀 레벨 IoU 요약 결과 출력
        print("픽셀 레벨 평가 (전체 Foreground IoU):")
        if summary['pixel_level_iou_scores']:
            print(f"  평균 Pixel-level IoU: {np.mean(summary['pixel_level_iou_scores']):.4f}")
        print()

        # 이미지별 상세 결과
        print("이미지별 상세 결과:")
        print("-" * 115)
        print(f"{'이미지':<20} {'GT':<5} {'Pred':<5} {'BBox_F1':<10} {'Mask_F1':<10} {'Combined_F1':<12} {'Pixel_IoU':<10}")
        print("-" * 115)
        
        for result in evaluation_results['individual_results']:
            print(f"{result['image_filename']:<20} {result['gt_count']:<5} {result['pred_count']:<5} "
                  f"{result['metrics']['bbox']['f1_score']:<10.4f} "
                  f"{result['metrics']['mask']['f1_score']:<10.4f} "
                  f"{result['metrics']['combined']['f1_score']:<12.4f} "
                  f"{result['metrics']['pixel_level_iou']:<10.4f}")
        
        print("=" * 115)
    
    def save_results_to_csv(self, evaluation_results: Dict[str, Any], output_path: str = "complete_evaluation_results.csv"):
        """결과를 CSV 파일로 저장"""
        results_data = []
        
        for result in evaluation_results['individual_results']:
            row = {
                'image_filename': result['image_filename'],
                'image_id': result['image_id'],
                'gt_count': result['gt_count'],
                'pred_count': result['pred_count'],
                'pixel_level_iou': result['metrics']['pixel_level_iou']
            }
            
            # 각 방식별 메트릭 추가
            for match_type in ['bbox', 'mask', 'combined']:
                metrics = result['metrics'][match_type]
                row.update({
                    f'{match_type}_precision': metrics['precision'],
                    f'{match_type}_recall': metrics['recall'],
                    f'{match_type}_f1_score': metrics['f1_score'],
                    f'{match_type}_true_positives': metrics['true_positives'],
                    f'{match_type}_false_positives': metrics['false_positives'],
                    f'{match_type}_false_negatives': metrics['false_negatives']
                })
            
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n결과가 {output_path}에 저장되었습니다.")
    
    def run_evaluation(self):
        """전체 평가 실행"""
        try:
            evaluation_results = self.evaluate_all_images()
            self.print_summary_results(evaluation_results)
            #self.save_results_to_csv(evaluation_results)
            return evaluation_results
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None