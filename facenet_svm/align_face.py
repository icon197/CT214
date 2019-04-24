import os
import cv2

if __name__ == "__main__":

    if not os.path.exists("./data_align"):
        os.makedirs("./data_align")

    for r_dir, s_dir, files in os.walk("./data"):
        # print(r_dir, s_dir, files)
        if len(s_dir) > 0:
            for path in s_dir:
                if not os.path.exists("./data_align/" + path):
                    os.makedirs("./data_align/" + path)
            continue
        for file in files:
            path_file = "/".join([r_dir, file])
            img = cv2.imread(path_file)
            img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('img', img)
            # x = cv2.waitKey(0)
            # if x == 27:
            #     cv2.destroyWindow('img')
            cv2.imwrite(path_file.replace("data", "data_align"), img)
