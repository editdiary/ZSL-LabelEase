# Zero-Shot Learning: LabelEase

## 프로젝트 개요

ZSL-LabelEase는 제로샷(Zero-Shot) 기반 딥러닝 모델을 활용하여 이미지 데이터의 자동 라벨링(탐지 및 분할) 성능을 검증하고, 실제 라벨링 업무의 적용 가능성을 실험하는 프로젝트입니다.

이 프로젝트는 세종대학교 `2025-1 AI로봇융합심화PBL 농업모듈` 대학원 강의 과제로 수행되었습니다.

## 논문 및 추가 자료

- 본 프로젝트와 관련된 자세한 실험 내용 및 결과는 첨부된 논문 파일([research_paper.pdf](./research_paper.pdf))에서 확인하실 수 있습니다.


## 배경 및 목적

딥러닝 모델의 성능을 높이기 위해서는 대량의 정밀한 라벨링 데이터가 필요합니다. 특히 classification → detection → segmentation으로 task가 복잡해질수록 라벨링에 필요한 시간과 노력이 기하급수적으로 증가합니다.

이러한 문제를 해결하기 위해 auto labeling 기능을 제공하는 다양한 도구들이 등장하고 있습니다.


## 실험 목표

- 제로샷 모델(Grounded SAM 2)을 활용해 기존에 학습하지 않은 객체(예: "참외")에 대한 탐지 및 분할 성능을 실험적으로 검증합니다.
- 사람이 직접 라벨링한 결과와 제로샷 모델이 자동으로 라벨링한 결과를 비교하여 annotation 보조 도구로서의 실용성을 평가합니다.


## 데이터

- 직접 수집한 "참외" 이미지 데이터셋 사용
- 기존 대형 모델이 학습하지 않은 특수 작물(참외)에 대해 제로샷 모델이 어느 정도까지 탐지/분할이 가능한지 확인


## 평가 방법

- **정량적 평가:**  IoU 등 지표로 수작업 라벨과 모델 라벨의 정확도 비교
- **정성적 평가:**  실제 결과 이미지를 직접 확인하며 모델의 한계와 가능성 해석


## 기대 효과

- 제로샷 모델을 활용한 annotation 보조 가능성 검증
- 데이터 구축 비용 및 시간 절감 효과 분석


## 환경 세팅 및 실행 방법

1. **Python 환경 준비**
   - conda를 이용해 Python 3.10 환경을 생성합니다.
     ```bash
     conda create -n pbl python=3.10
     ```
   - 생성한 환경을 활성화합니다.
     ```bash
     conda activate pbl
     ```

2. **모델 체크포인트 다운로드**
   - SAM2 가중치 다운로드:
     ```bash
     cd GroundedSAM2/checkpoints
     bash download_ckpts.sh
     ```
   - Grounding DINO 가중치 다운로드:
     ```bash
     cd ../gdino_checkpoints
     bash download_ckpts.sh
     ```

3. **PyTorch 및 관련 패키지 설치**

   - PyTorch 2.3.1, torchvision 0.18.1, torchaudio 2.3.1을 설치합니다.  
     (CUDA 12.1을 지원하는 버전 기준)
     ```bash
     pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
     ```
   - ⚠️ **주의:**  
     위 명령어는 CUDA 12.1 환경에 맞는 버전입니다.  
     사용자의 CUDA 버전에 따라 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)에서 알맞은 설치 명령어를 확인하여 조정해 주세요.  
     [Note!] 코드를 실행하기 위해서는 **최소 CUDA 12.1 이상**이 필요합니다.

4. **CUDA 환경 변수 설정**

   - 자신의 CUDA 설치 경로에 맞게 환경 변수를 설정해야 합니다.
     ```bash
     export CUDA_HOME=/path/to/cuda-12.1/
     ```
   - 예를 들어, 기본 경로에 설치된 경우:
     ```bash
     export CUDA_HOME=/usr/local/cuda-12.1/
     ```

5. **로컬 패키지 설치**

   - SAM2 패키지 설치:
     ```bash
     pip install -e .
     ```
     (프로젝트 루트 디렉토리에서 실행)

   - Grounding DINO 패키지 설치:
     ```bash
     pip install --no-build-isolation -e grounding_dino
     ```
     (마찬가지로 프로젝트 루트에서 실행)

6. **기타 패키지 설치**

   - 코드 실행 중 추가로 필요한 패키지가 있을 수 있습니다.
   - 오류 메시지에 따라 필요한 패키지를 아래와 같이 설치해 주세요.
     ```bash
     pip install 패키지명
     ```

7. **데이터셋 준비**

   - 실험에 사용할 데이터셋은 `GroundedSAM2/my_dataset` 폴더에 준비해야 합니다.
   - 예시 폴더 구조:
     ```
     GroundedSAM2/my_dataset/
       ├── rfv/
       │    ├── images/
       │    └── label/
       └── bfv/
            ├── images/
            └── label/
     ```
   - 각 하위 폴더(`rfv`, `bfv` 등)에는 `images`(이미지 파일)와 `label`(수동 라벨 파일) 디렉토리를 포함해야 합니다.
   - 실험에 사용할 데이터 파일들을 해당 위치에 넣어주세요.

8. **코드 실행 방법**

   - 아래 명령어로 실행 디렉토리로 이동합니다.
     ```bash
     cd GroundedSAM2
     ```
   - 코드를 실행합니다.
     ```bash
     python mydata_local.py
     ```


## 참고

- Zero-Shot 모델: [Grounded SAM 2 (GitHub)](https://github.com/IDEA-Research/Grounded-SAM-2)
- 본 프로젝트는 연구 및 실험 목적입니다.


## Contact
E-mail: dhlee@sju.ac.kr
