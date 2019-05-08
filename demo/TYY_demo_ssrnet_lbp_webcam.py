import cv2
import numpy as np
from SSRNET_model import SSR_net, SSR_net_general
import timeit
import time
from mtcnn.mtcnn import MTCNN
# import os


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def show_results(detected, input_img, faces, ad, img_size, img_w, img_h, model, model_gender, time_detection, time_network, time_plot, mtcnn):

    draw = False

    if mtcnn:
        detected = list(map(lambda x: x['box'], detected))

    for i, (x, y, w, h) in enumerate(detected):
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        if draw:
            cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

    start_time = timeit.default_timer()
    if len(detected) > 0:
        # predict ages and genders of the detected faces
        predicted_ages = model.predict(faces)
        predicted_genders = model_gender.predict(faces)
    elapsed_time = timeit.default_timer()-start_time
    time_network = time_network + elapsed_time

    # draw results
    print('===========DETECTION===========')

    for i, (x, y, w, h) in enumerate(detected):
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        gender_str = 'M'
        if predicted_genders[i] < 0.5:
            gender_str = 'F'

        label = "{},{}".format(int(predicted_ages[i]), gender_str)
        if draw:
            draw_label(input_img, (x1, y1), label)
        print(label)
    print('==============END==============')

    start_time = timeit.default_timer()
    if draw:
        cv2.imshow("result", input_img)
    elapsed_time = timeit.default_timer()-start_time
    time_plot = time_plot + elapsed_time

    return input_img, time_network, time_plot


def main():
    weight_file = "../pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "../pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    mtccn = True

    if mtccn:
        detector = MTCNN()
    else:
        detector = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)
    model_gender = SSR_net_general(img_size, stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(weight_file_gender)

    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)

    detected = ''
    time_detection = 0
    time_network = 0
    time_plot = 0
    ad = 0.5
    sleep = 0
    img_idx = 0
    skip_frame = 10

    while True:
        if sleep:
            time.sleep(sleep/1000)
        # get video frame
        img_idx = img_idx + 1
        ret, input_img = cap.read()
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0
            # detect faces using LBP detector
            start_time = timeit.default_timer()
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            if mtccn:
                detected = detector.detect_faces(input_img)
            else:
                detected = detector.detectMultiScale(gray_img, 1.1)
            elapsed_time = timeit.default_timer()-start_time
            time_detection = time_detection + elapsed_time
            faces = np.empty((len(detected), img_size, img_size, 3))

        input_img, time_network, time_plot = show_results(
            detected,
            input_img,
            faces,
            ad,
            img_size,
            img_w,
            img_h,
            model,
            model_gender,
            time_detection,
            time_network,
            time_plot,
            mtccn
        )

        # Show the time cost (fps)
        # print('time_detection:', time_detection)
        # print('time_network:', time_network)
        # print('time_plot:', time_plot)
        # print('===============================')
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
