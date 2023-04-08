import cv2
import tensorflow.keras
import numpy as np
from playsound import playsound
#class1 = 마스크 + 모자 착용
#class2 = 마스크만 착용

## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

## 학습된 모델 불러오기
model_filename = 'keras_model.h5' ####티처블 머신으로 교육한 파일 넣는곳!!!!!!
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

put_cnt = 1 # 안전모 착용 상태를 확인하기 위한 변수 
while True: # 특정 키를 누를 때까지 무한 반복
    ret, frame = capture.read() # 한 프레임씩 읽기
    if ret == True: 
        print("read success!")

    # 이미지 좌우반전시키기
    frame_fliped = cv2.flip(frame, 1)
    
    # 이미지 출력하는 프레임 실행
    cv2.imshow("VideoFrame", frame_fliped)
    
    # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
    if cv2.waitKey(200) > 0: 
        break
    
    # 데이터 전처리
    preprocessed = preprocessing(frame_fliped)

    # 예측
    # prediction[0,0]은 class1의 확률을 의미하고, prediction[0,1]은 class2의 확률을 의미함.
    #즉, class2의  확률값이 크면 안전모 착용중인 상태를 인지
    prediction = model.predict(preprocessed)
    #print(prediction) # [[0.00533728 0.99466264]]
    
    if prediction[0,0] < prediction[0,1]: #안전모를 착용한 경우
        print('안전모 착용')
        put_cnt += 1
        
    else: #안전모를 착용하지 않은 경우(class2) "안전모를 착용하십시오" 음성 출력
        print('--위험--')
        put_cnt = 1
        playsound("audio2.mp3")
       
    
# 카메라 객체 반환
capture.release() 
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()