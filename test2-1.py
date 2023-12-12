import cv2
import numpy as np

## 변경할 이미지 모델 가져오기
net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
net2 = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')

img = cv2.imread('imgs/hw.jpg')
print(img.shape)

#같은 비율로 사이즈 줄이기
h, w, c = img.shape
img = cv2.resize(img, dsize=(1024, int(h / w * 1024)))
print(img.shape)

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

net2.setInput(blob)
output2 = net2.forward()
output2 = output2.squeeze().transpose((1, 2, 0))
output2 += MEAN_VALUE
output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

#가로로 반으로 자르기
output = output[0:288,:]
output2 = output2[288:576,:]

#axis=1 y방향으로 합치기
output3 = np.concatenate([output,output2], axis=0)

cv2.imshow('img', img)
# cv2.imshow('result', output)
# cv2.imshow('result2', output2)
cv2.imshow('result3', output3)
cv2.waitKey(0)