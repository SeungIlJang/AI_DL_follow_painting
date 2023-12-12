import cv2
import numpy as np

# net = cv2.dnn.readNetFromTorch('models/eccv16/starry_night.t7')
net = cv2.dnn.readNetFromTorch('models/instance_norm/candy.t7')

# img = cv2.imread('imgs/01.jpg')
img = cv2.imread('imgs/02.jpg')
print(img.shape)
#높이 너비 채널 : 이미지 형태
h, w, c = img.shape
#비율에 맞게 리사이징
img = cv2.resize(img, dsize=(500, int(h / w * 500)))

#이미지 자르기
img = img[162:513, 185:428]

print(img.shape)
#전처리값,이값이 커피를 맛있게함
MEAN_VALUE = [103.939, 116.779, 123.680]
#전처리, 차원변형을 통해 컴퓨터가 알수 있게 변경
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

print(blob.shape)

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

cv2.imshow('output',output)
cv2.imshow('img',img)
cv2.waitKey(0)
