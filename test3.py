import cv2
import numpy as np

## 변경할 이미지 모델 가져오기
net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')

#동영상 불러오기
cap = cv2.VideoCapture('mp4/03.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    #같은 비율로 사이즈 줄이기
    h, w, c = img.shape
    img = cv2.resize(img, dsize=(640, int(h / w * 640)))
    # print(img.shape)    
    
    #전처리값,가장 효과가 좋은 값
    MEAN_VALUE = [103.939, 116.779, 123.680]
    #전처리, 차원변형을 통해 컴퓨터가 알수 있게 변경
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    #결과 추론하기(Inference)
    net.setInput(blob)
    output = net.forward()

    #후처리 차원 줄이기-squeeze 순서 바꾸기-transpose
    output = output.squeeze().transpose((1, 2, 0))
    #MEAN_VALUE 더하기 전처리 했을 때 MEAN_VALUE 를 뺐는데 후처리 할 때는 MEAN_VALUE를 다시 더해줌
    output += MEAN_VALUE
    #범위 넘어가는 값 잘라내기 clip
    output = np.clip(output, 0, 255)
    #자료형 바꾸기 astype
    output = output.astype('uint8')
        
    cv2.imshow('result', output)

    if cv2.waitKey(1) == ord('q'):
        break


