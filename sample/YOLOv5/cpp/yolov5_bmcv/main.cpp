//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "yolov5.hpp"
using json = nlohmann::json;
using namespace std;

extern "C" {
  struct YOLOResult {
    int x, y, width, height;
    float score;
    int class_id;
  };

  struct Result {
    YOLOResult* data;
    int num_dets;
  };
  
  void free_results(YOLOResult* results) {
    delete[] results;
  }

  Result detect(const char* str1, unsigned char* image, int row, int col){
    cout.setf(ios::fixed);
    string bmodel_file(str1);
    // 传入img 必须复制一次，不然所在内存不一致，无法转换BMimg
    cv::Mat img(row, col, CV_8UC3, (void *)image);
    cv::Mat copy_img;
    img.copyTo(copy_img);
    bm_image bmimg;
    cv::bmcv::toBMI(copy_img, &bmimg, true);
    
    int dev_id = 0;
    float conf_thresh = 0.25;
    float nms_thresh = 0.4;
    string coco_names = "123";

    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    bm_handle_t h = handle->handle();
    // load bmodel
    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());
    // initialize net
    YoloV5 yolov5(bm_ctx);
    CV_Assert(0 == yolov5.Init(conf_thresh, nms_thresh, coco_names));
    // profiling
    // TimeStamp yolov5_ts;
    // TimeStamp *ts = &yolov5_ts;
    // yolov5.enableProfile(&yolov5_ts);
    // get batch_size
    int batch_size = yolov5.batch_size();
    // creat save path
    // if (access("results", 0) != F_OK)
    //   mkdir("results", S_IRWXU);
    // if (access("results/images", 0) != F_OK)
    //   mkdir("results/images", S_IRWXU);

    vector<bm_image> batch_imgs;
    vector<YoloV5BoxVec> boxes;
    // test images
    if (dev_id == 0){
      batch_imgs.push_back(bmimg);

      if ((int)batch_imgs.size() == batch_size){
        // predict
        CV_Assert(0 == yolov5.Detect(batch_imgs, boxes));
        for(int i = 0; i < batch_size; i++){
          if (batch_imgs[i].image_format != 0){
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width, FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }
          // for (auto bbox : boxes[i]) {
          //   cout << "  class id=" << bbox.class_id << ", score = " << bbox.score << " (x=" << bbox.x << ",y=" << bbox.y << ",w=" << bbox.width << ",h=" << bbox.height << ")" << endl;
            // draw image
          //   yolov5.draw_bmcv(h, bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.width, bbox.height, batch_imgs[i]);
          //   cout << "-------------succesed draw image --------------" << endl;
          // }
          // save image
          // void* jpeg_data = NULL;
          // size_t out_size = 0;
          // int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
          // if (ret == BM_SUCCESS) {
          //   string img_file = "results/images/1.jpg";
          //   FILE *fp = fopen(img_file.c_str(), "wb");
          //   fwrite(jpeg_data, out_size, 1, fp);
          //   fclose(fp);
          // }
          // free(jpeg_data);
          // cout << "-------------succesed write image --------------" << endl;
          bm_image_destroy(batch_imgs[i]);
          img.release();
          copy_img.release();
        }
      }
    }
    batch_imgs.clear();
    int num_results = boxes.size();
    cout << "detect num :" << num_results << endl;
    std::vector<YOLOResult> res;
    for (const auto& detection : boxes[0]) {
      YOLOResult t;
      t.x = detection.x;
      t.y = detection.y;
      t.width = detection.width;
      t.height = detection.height;    
      t.score = detection.score;
      t.class_id = (int)detection.class_id;
      res.push_back(t);
    }

    Result result;
    result.num_dets = num_results;
    result.data = new YOLOResult[res.size()];
    std::copy(res.begin(), res.end(), result.data);
    
    boxes.clear();
    // print speed
    // time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    // yolov5_ts.calbr_basetime(base_time);
    // yolov5_ts.build_timeline("yolov5 test");
    // yolov5_ts.show_summary("yolov5 test");
    // yolov5_ts.clear();
    // for (int i = 0; i < result.num_dets; i++) {
    //   YOLOResult& yoloResult = result.data[i];
    //   std::cout << "Detection " << i + 1 << ":" << std::endl;
    //   std::cout << "x: " << yoloResult.x << std::endl;
    //   std::cout << "y: " << yoloResult.y << std::endl;
    //   std::cout << "width: " << yoloResult.width << std::endl;
    //   std::cout << "height: " << yoloResult.height << std::endl;
    //   std::cout << "score: " << yoloResult.score << std::endl;
    //   std::cout << "class_id: " << yoloResult.class_id << std::endl;
    //   std::cout << std::endl;
    // }
    return result;
  }
}