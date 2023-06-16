import ctypes
import sophon.sail as sail
import cv2

# mkdir build && cd build
# # 请根据实际情况修改-DSDK的路径，需使用绝对路径。
# cmake -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-0.4.6/ ..
# make

# struct YoloV5Box {
#   int x, y, width, height;
#   float score;
#   int class_id;
# };

# 定义自定义结构体
class YOLOResult(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_int),
        ('y', ctypes.c_int),
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('score', ctypes.c_float),
        ('class_id', ctypes.c_int),
    ]

class Result(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(YOLOResult)),
        ('num_dets', ctypes.c_int),
    ]

#img_file = '/data/sophon-demo/sample/YOLOv5/pics/zidane_cpp_bmcv.jpg'
img_file = '/data/sophon-demo/sample/YOLOv5/pics/zidane_python_opencv.jpg'
opencv_img = cv2.imread(img_file)
rows, cols = (opencv_img.shape[0], opencv_img.shape[1])
opencv_mat = opencv_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

# model_path = '/data/sophon-demo/sample/YOLOv5/cpp/person_detector_int8_1b.bmodel'
decoder = sail.Decoder(img_file, True, 0)
bmimg = sail.BMImage()
handle = sail.Handle(0)
ret = decoder.read(handle, bmimg)
# print(bmimg.asmat().shape)
# bmcv = sail.Bmcv(handle)
# bmcv.imwrite('/data/sophon-demo/sample/YOLOv5/cpp/yolov5_bmcv/test.jpg', bmimg)

# 加载共享库
lib = ctypes.cdll.LoadLibrary('/data/sophon-demo/sample/YOLOv5/cpp/yolov5_bmcv/libyolov5_bmcv.so')

# 指定函数参数和返回类型
lib.detect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
lib.detect.restype = Result

# 调用C++函数
img_file = b"/data/sophon-demo/sample/YOLOv5/pics/zidane_cpp_bmcv.jpg"
model_path = b"/data/sophon-demo/sample/YOLOv5/cpp/person_detector_int8_1b.bmodel"
# model_path = b"/data/sophon-demo/sample/YOLOv5/cpp/fire_smog_int8_1b.bmodel"
result = lib.detect(model_path, opencv_mat, rows, cols)
print(f"___________________________")
print(f"num_dets :{result.num_dets}")
results = [result.data[i] for i in range(result.num_dets)]
# 打印结果
for item in results:
    print(item.class_id, item.score)
    print(item.x, item.y, item.width, item.height, item.class_id, item.score)

# 释放内存
lib.free_results(result.data)
# # mylib.releaseCustomVectorVector(result)
