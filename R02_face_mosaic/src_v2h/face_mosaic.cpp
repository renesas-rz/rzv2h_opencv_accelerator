/*
 * Original Code (C) Copyright Edgecortix, Inc. 2022
 * Modified Code (C) Copyright Renesas Electronics Corporation 2023-2024
 *　
 *  *1 DRP-AI TVM is powered by EdgeCortix MERA(TM) Compiler Framework.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

/***********************************************************************************************************************
* File Name    : head_count_topview.cpp
* Version      : 1.1.0
* Description  : DRP-AI TVM[*1] Application Example
***********************************************************************************************************************/

/*****************************************
* includes
******************************************/
#include "define.h"
#include "box.h"
#include "MeraDrpRuntimeWrapper.h"
#include <linux/drpai.h>
#include <linux/input.h>
#include <builtin_fp16.h>
#include <opencv2/opencv.hpp>
#include "wayland.h"

/*Pre-processing Runtime Header*/
#include "PreRuntime.h"
/*dmabuf for Pre-processing Runtime input data*/
#include "dmabuf.h"

using namespace std;
using namespace cv;

/* DRP-AI TVM[*1] Runtime object */
MeraDrpRuntimeWrapper runtime;
/* Pre-processing Runtime object */
PreRuntime preruntime;
/*MMNGR buffer for DRP-AI Pre-processing*/
static dma_buffer *drpai_buf;

/*Global Variables*/
static float drpai_output_buf[INF_OUT_SIZE];
static Wayland wayland;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
static pthread_t memcpy_thread;

/*Flags*/
static std::atomic<uint8_t> capture_start           (1);
static std::atomic<uint8_t> inference_start         (0);
static std::atomic<uint8_t> img_processing_start    (0);

static sem_t terminate_req_sem;
static int32_t drp_max_freq;
static int32_t drp_freq;

static atomic<uint8_t> hdmi_obj_ready   (0);

static uint32_t disp_time = 0;

static int box_mode;
static int flip_mode;

std::string media_port;
std::string gstreamer_pipeline;

std::vector<float> floatarr(1);

uint64_t drpaimem_addr_start = 0;
bool runtime_status = false; 
static vector<detection> det;
static vector<detection> det_sub;

float fps = 0;
float CAPTURE_TIME = 0;
float CAPTURE_WAIT_TIME = 0;
float MEMCPY_TIME = 0;
float TOTAL_TIME = 0;
float PRE_PROC_TIME = 0;
float INF_TIME= 0;
float POST_PROC_TIME = 0;
float INF_WAIT_TIME = 0;
float IMAGE_TIME = 0;
float IMAGE_WAIT_TIME = 0;

int32_t HEAD_COUNT= 0;
int fd;

/*Global frame */
Mat g_frame;
VideoCapture cap;

cv::Mat inf_frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
cv::Mat img_frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

/* Map to store input source list */
std::map<std::string, int> input_source_map ={    
    {"USB", 1}
     } ;


/*****************************************
 * Function Name     : float16_to_float32
 * Description       : Function by Edgecortex. Cast uint16_t a into float value.
 * Arguments         : a = uint16_t number
 * Return value      : float = float32 number
 ******************************************/
float float16_to_float32(uint16_t a)
{
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

/*****************************************
 * Function Name     : load_label_file
 * Description       : Load label list text file and return the label list that contains the label.
 * Arguments         : label_file_name = filename of label list. must be in txt format
 * Return value      : vector<string> list = list contains labels
 *                     empty if error occurred
 ******************************************/
vector<string> load_label_file(string label_file_name)
{
    vector<string> list = {};
    vector<string> empty = {};
    ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        return list;
    }

    string line = "";
    while (getline(infile, line))
    {
        list.push_back(line);
        if (infile.fail())
        {
            return empty;
        }
    }

    return list;
}

/*****************************************
 * Function Name : sigmoid
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : x = input argument for the calculation
 * Return value  : sigmoid result of input x
 ******************************************/
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/*****************************************
* Function Name : yolo_index
* Description   : Get the index of the bounding box attributes based on the input offset
* Arguments     : n = output layer number
                  offs = offset to access the bounding box attributes
*                 channel = channel to access each bounding box attribute.
* Return value  : index to access the bounding box attribute.
******************************************/
int32_t yolo_index(uint8_t n, int32_t offs, int32_t channel)
{
    uint8_t num_grid = num_grids[n];
    return offs + channel * num_grid * num_grid;
}

/*****************************************
* Function Name : yolo_offset
* Description   : Get the offset number to access the bounding box attributes
*                 To get the actual value of bounding box attributes, use yolo_index() after this function.
* Arguments     : n = output layer number [0~2].
                  b = Number to indicate which bounding box in the region [0~4]
*                 y = Number to indicate which region [0~13]
*                 x = Number to indicate which region [0~13]
* Return value  : offset to access the bounding box attributes.
******************************************/
int32_t yolo_offset(uint8_t n, int32_t b, int32_t y, int32_t x)
{
    uint8_t num = num_grids[n];
    uint32_t prev_layer_num = 0;
    int32_t i = 0;

    for (i = 0; i < n; i++)
    {
        prev_layer_num += NUM_BB * (NUM_CLASS + 5) * num_grids[i] * num_grids[i];
    }
    return prev_layer_num + b * (NUM_CLASS + 5) * num * num + y * num + x;
}


static int8_t wait_join(pthread_t *p_join_thread, uint32_t join_time)
{
    int8_t ret_err;
    struct timespec join_timeout;
    ret_err = clock_gettime(CLOCK_REALTIME, &join_timeout);
    if ( 0 == ret_err )
    {
        join_timeout.tv_sec += join_time;
        ret_err = pthread_timedjoin_np(*p_join_thread, NULL, &join_timeout);
    }
    return ret_err;
}

/*****************************************
 * Function Name : R_Post_Proc
 * Description   : Process CPU post-processing for YOLOv3
 * Arguments     : floatarr = drpai output address
 * Return value  : -
 ******************************************/
void R_Post_Proc(float *floatarr)
{
    /* Following variables are required for correct_region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    float new_w, new_h;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float)(MODEL_IN_W / correct_w) < (float)(MODEL_IN_H / correct_h))
    {
        new_w = (float)MODEL_IN_W;
        new_h = correct_h * MODEL_IN_W / correct_w;
    }
    else
    {
        new_w = correct_w * MODEL_IN_H / correct_h;
        new_h = MODEL_IN_H;
    }

    int32_t n = 0;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;
    int32_t i = 0;
    float tx = 0;
    float ty = 0;
    float tw = 0;
    float th = 0;
    float tc = 0;
    float center_x = 0;
    float center_y = 0;
    float box_w = 0;
    float box_h = 0;
    float objectness = 0;
    uint8_t num_grid = 0;
    uint8_t anchor_offset = 0;
    float classes[NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    detection d;
    /* Clear the detected result list */
    det.clear();

    /*Post Processing Start*/
    for (n = 0; n < NUM_INF_OUT_LAYER; n++)
    {
        num_grid = num_grids[n];
        anchor_offset = 2 * NUM_BB * (NUM_INF_OUT_LAYER - (n + 1));

        for (b = 0; b < NUM_BB; b++)
        {
            for (y = 0; y < num_grid; y++)
            {
                for (x = 0; x < num_grid; x++)
                {
                    offs = yolo_offset(n, b, y, x);
                    tx = floatarr[offs];
                    ty = floatarr[yolo_index(n, offs, 1)];
                    tw = floatarr[yolo_index(n, offs, 2)];
                    th = floatarr[yolo_index(n, offs, 3)];
                    tc = floatarr[yolo_index(n, offs, 4)];
                    /* Compute the bounding box */
                    /*get_region_box*/
                    center_x = ((float)x + sigmoid(tx)) / (float)num_grid;
                    center_y = ((float)y + sigmoid(ty)) / (float)num_grid;
                    box_w = (float)exp(tw) * anchors[anchor_offset + 2 * b + 0] / (float)MODEL_IN_W;
                    box_h = (float)exp(th) * anchors[anchor_offset + 2 * b + 1] / (float)MODEL_IN_W;
                    /* Adjustment for VGA size */
                    /* correct_region_boxes */
                    center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float)new_w / MODEL_IN_W);
                    center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float)new_h / MODEL_IN_H);
                    box_w *= (float)(MODEL_IN_W / new_w);
                    box_h *= (float)(MODEL_IN_H / new_h);
                    center_x = round(center_x * DRPAI_IN_WIDTH);
                    center_y = round(center_y * DRPAI_IN_HEIGHT);
                    box_w = round(box_w * DRPAI_IN_WIDTH);
                    box_h = round(box_h * DRPAI_IN_HEIGHT);
                    objectness = sigmoid(tc);
                    Box bb = {center_x, center_y, box_w, box_h};
                    /* Get the class prediction associated with each BB  [5: ] */
                    for (i = 0; i < NUM_CLASS; i++)
                    {
                        classes[i] = sigmoid(floatarr[yolo_index(n, offs, 5 + i)]);
                    }
                    max_pred = 0;
                    pred_class = -1;
                    /*Get the predicted class */
                    for (i = 0; i < NUM_CLASS; i++)
                    {
                        if (classes[i] > max_pred)
                        {
                            pred_class = i;
                            max_pred = classes[i];
                        }
                    }

                    /* Store the result into the list if the probability is more than the threshold */
                    probability = max_pred * objectness;
                    if (probability > TH_PROB)
                    {
                        d = {bb, pred_class, probability};
                        det.push_back(d);
                    }
                }
            }
        }
    }

    /* Non-Maximum Suppression filter */
    filter_boxes_nms(det, det.size(), TH_NMS);
    
    return;
}

/*****************************************
 * Function Name : draw_bounding_box
 * Description   : Draw bounding box on image.
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
void draw_bounding_box(void)
{
    stringstream stream;
    string str = "";
    string result_str;
    int32_t result_cnt =0;
    uint32_t x = HEAD_COUNT_STR_X;
    uint32_t y = HEAD_COUNT_STR_X;
    static int32_t mosaic_rate = 0;

    static int32_t x_mosaic = 0;
    static int32_t y_mosaic = 0;
    static int32_t x_mosaic_div = 0;
    static int32_t y_mosaic_div = 0;

    unsigned long OCA_f[16];

    /* Draw bounding box on RGB image. */
    int32_t i = 0;
    for (i = 0; i < det_sub.size(); i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det_sub[i].prob == 0)
        {
            continue;
        }
        result_cnt++;
        /* Clear string stream for bounding box labels */
        stream.str("");
        /* Draw the bounding box on the image */
        stream << fixed << setprecision(2) << det_sub[i].prob;
        result_str = label_file_map[det_sub[i].c] + " " + stream.str();
           
        int32_t x_min = (int)det_sub[i].bbox.x - round((int)det_sub[i].bbox.w / 2.);
        int32_t y_min = (int)det_sub[i].bbox.y - round((int)det_sub[i].bbox.h / 2.);
        int32_t x_max = (int)det_sub[i].bbox.x + round((int)det_sub[i].bbox.w / 2.) - 1;
        int32_t y_max = (int)det_sub[i].bbox.y + round((int)det_sub[i].bbox.h / 2.) - 1;

        /* Check the bounding box is in the image range */
        x_min = x_min < 1 ? 1 : x_min;
        x_max = ((BOX_IN_WIDTH - 2) < x_max) ? (BOX_IN_WIDTH - 2) : x_max;
        y_min = y_min < 1 ? 1 : y_min;
        y_max = ((BOX_IN_HEIGHT - 2) < y_max) ? (BOX_IN_HEIGHT - 2) : y_max;

        int32_t x2_min = x_min + BOX_THICKNESS;
        int32_t y2_min = y_min + BOX_THICKNESS;
        int32_t x2_max = x_max - BOX_THICKNESS;
        int32_t y2_max = y_max - BOX_THICKNESS;

        x2_min = ((BOX_IN_WIDTH - 2) < x2_min) ? (BOX_IN_WIDTH - 2) : x2_min;
        x2_max = x2_max < 1 ? 1 : x2_max;
        y2_min = ((BOX_IN_HEIGHT - 2) < y2_min) ? (BOX_IN_HEIGHT - 2) : y2_min;
        y2_max = y2_max < 1 ? 1 : y2_max;

        x_mosaic = x_max - x_min - 2;   // width size of the mosaic area
        y_mosaic = y_max - y_min - 2;   // height size of the mosaic area

        if((x_mosaic + y_mosaic) < 300)
        {
            mosaic_rate = 10;
        }
        else
        {
            mosaic_rate = 20;
        }

        x_mosaic_div = x_mosaic / mosaic_rate;
        y_mosaic_div = y_mosaic / mosaic_rate;

        x_mosaic_div = x_mosaic_div < 2 ? 2 : x_mosaic_div;
        y_mosaic_div = y_mosaic_div < 2 ? 2 : y_mosaic_div;

        // CPU Mosaic
        cv::Mat mid_image(x_mosaic_div, y_mosaic_div, CV_8UC3);
        //BGR image Crop
        cv::Mat crop_image = cv::Mat(img_frame, cv::Rect(x_min + 1, y_min + 1, x_mosaic, y_mosaic)).clone();

        OCA_f[DRP_FUNC_RESIZE] = OPENCVA_FUNC_ENABLE;  //OCA
        OCA_Activate(&OCA_f[0]);

        cv::resize(crop_image, mid_image, {x_mosaic_div, y_mosaic_div}, 0, 0, cv::INTER_LINEAR);   // reduction

        OCA_f[DRP_FUNC_RESIZE] = OPENCVA_FUNC_DISABLE;  //CPU
        OCA_Activate(&OCA_f[0]);

        cv::resize(mid_image, crop_image, {x_mosaic, y_mosaic}, 0, 0, cv::INTER_NEAREST);   // expansion

        crop_image.copyTo(img_frame(Rect(x_min + 1, y_min + 1, x_mosaic, y_mosaic)));

        if(box_mode == 1)
        {
            Point topLeft(x_min, y_min);
            Point bottomRight(x_max, y_max);

            Point topLeft2(x2_min, y2_min);
            Point bottomRight2(x2_max, y2_max);

            /* Creating bounding box and class labels */
            /*cordinates for solid rectangle*/
            Point textleft(x_min,y_min+CLASS_LABEL_HEIGHT);
            Point textright(x_min+CLASS_LABEL_WIDTH,y_min);

            rectangle(img_frame, topLeft, bottomRight, Scalar(0, 0, 0), BOX_THICKNESS);
            rectangle(img_frame, topLeft2, bottomRight2, Scalar(255, 255, 255), BOX_THICKNESS);
            /*solid rectangle over class name */
            rectangle(img_frame, textleft, textright, Scalar(59, 94, 53), -1);
            putText(img_frame, result_str, textleft, FONT_HERSHEY_SIMPLEX, CHAR_SCALE_XS, Scalar(255, 255, 255), BOX_CHAR_THICKNESS);
        }
    }
    HEAD_COUNT = result_cnt++;
    return;
}

/*****************************************
 * Function Name : Head Detection
 * Description   : Function to perform over all detection
 * Arguments     : -
 * Return value  : 0 if succeeded
 *               not 0 otherwise
 ******************************************/
int Head_Detection()
{   
    /*Variable for getting Inference output data*/
    void* output_ptr;
    uint32_t out_size;

    /*Variable for Pre-processing parameter configuration*/
    s_preproc_param_t in_param;

    /*Variable for checking return value*/
    int8_t ret = 0;

    in_param.pre_in_shape_w = DRPAI_IN_WIDTH;
    in_param.pre_in_shape_h = DRPAI_IN_HEIGHT;

    /* Preprocess time start */
    auto t0 = std::chrono::high_resolution_clock::now();

    in_param.pre_in_addr = (uintptr_t) drpai_buf->phy_addr;

    ret = preruntime.Pre(&in_param, &output_ptr, &out_size);

    if (0 < ret)
    {
        fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Pre()\n");
        return 0;
    }

    /* Preprocess time ends*/
    auto t1 = std::chrono::high_resolution_clock::now();

    /*Set Pre-processing output to be inference input. */
    runtime.SetInput(0, (float*)output_ptr);

    /* Inference time start */
    auto t2 = std::chrono::high_resolution_clock::now();
    runtime.Run();
    /* Inference time end */
    auto t3 = std::chrono::high_resolution_clock::now();
    auto inf_duration = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    /* Postprocess time start */
    auto t4 = std::chrono::high_resolution_clock::now();

    /*load inference out on drpai_out_buffer*/
    int32_t i = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void *, int64_t> output_buffer;
    int64_t output_size;
    uint32_t size_count = 0;

    /* Get the number of output of the target model. */
    output_num = runtime.GetNumOutput();

    size_count = 0;
    /*GetOutput loop*/
    for (i = 0; i < output_num; i++)
    {
        /* output_buffer below is tuple, which is { data type, address of output data, number of elements } */
        output_buffer = runtime.GetOutput(i);
        /*Output Data Size = std::get<2>(output_buffer). */
        output_size = std::get<2>(output_buffer);

        /*Output Data Type = std::get<0>(output_buffer)*/
        if (InOutDataType::FLOAT16 == std::get<0>(output_buffer))
        {
            /*Output Data = std::get<1>(output_buffer)*/
            uint16_t *data_ptr = reinterpret_cast<uint16_t *>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++)
            {
                /*FP16 to FP32 conversion*/
                drpai_output_buf[j + size_count] = float16_to_float32(data_ptr[j]);
            }
        }
        else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer))
        {
            /*Output Data = std::get<1>(output_buffer)*/
            float *data_ptr = reinterpret_cast<float *>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++)
            {
                drpai_output_buf[j + size_count] = data_ptr[j];
            }
        }
        else
        {
            std::cerr << "[ERROR] Output data type : not floating point." << std::endl;
            ret = -1;
            break;
        }
        size_count += output_size;
    }
    if (ret != 0)
    {
        std::cerr << "[ERROR] DRP Inference Not working !!! " << std::endl;
        return -1;
    }

   /* Do post process to get bounding boxes */
    R_Post_Proc(drpai_output_buf);

    /* Postprocess time end */
    auto t5 = std::chrono::high_resolution_clock::now();
    
    auto r_post_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto pre_proc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    POST_PROC_TIME = r_post_proc_time/1000.0;
    PRE_PROC_TIME = pre_proc_time/1000.0;
    INF_TIME = inf_duration/1000.0;
    float total_time = float(inf_duration/1000.0) + float(POST_PROC_TIME) + float(pre_proc_time/1000.0);
    TOTAL_TIME = total_time;
    return 0;
}

void click_event(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDBLCLK)
    {
        std::cout<<"Exiting the event "<<std::endl;
        exit(0);
    }
}

/*****************************************
* Function Name : get_drpai_start_addr
* Description   : Function to get the start address of DRPAImem.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/
uint64_t get_drpai_start_addr(int drpai_fd)
{
    int ret = 0;
    drpai_data_t drpai_data;

    errno = 0;

    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd , DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (-1 == ret)
    {
        std::cerr << "[ERROR] Failed to get DRP-AI Memory Area : errno=" << errno << std::endl;
        return 0;
    }

    return drpai_data.address;
}

/*****************************************
* Function Name : set_drpai_freq
* Description   : Function to set the DRP and DRP-AI frequency.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int set_drpai_freq(int drpai_fd)
{
    int ret = 0;
    uint32_t data;

    errno = 0;
    data = drp_max_freq;
    ret = ioctl(drpai_fd , DRPAI_SET_DRP_MAX_FREQ, &data);
    if (-1 == ret)
    {
        std::cerr << "[ERROR] Failed to set DRP Max Frequency : errno=" << errno << std::endl;
        return -1;
    }

    errno = 0;
    data = drp_freq;
    ret = ioctl(drpai_fd , DRPAI_SET_DRPAI_FREQ, &data);
    if (-1 == ret)
    {
        std::cerr << "[ERROR] Failed to set DRP-AI Frequency : errno=" << errno << std::endl;
        return -1;
    }

    return 0;
}

/*****************************************
* Function Name : init_drpai
* Description   : Function to initialize DRP-AI.
* Arguments     : drpai_fd: DRP-AI file descriptor
* Return value  : If non-zero, DRP-AI memory start address.
*                 0 is failure.
******************************************/
uint64_t init_drpai(int drpai_fd)
{
    int ret = 0;
    uint64_t drpai_addr = 0;

    /*Get DRP-AI memory start address*/
    drpai_addr = get_drpai_start_addr(drpai_fd);
    if (drpai_addr == 0)
    {
        return 0;
    }

    /*Set DRP-AI frequency*/
    ret = set_drpai_freq(drpai_fd);
    if (ret != 0)
    {
        return 0;
    }

    return drpai_addr;
}

/*****************************************
 * Function Name : query_device_status
 * Description   : function to check USB device is connected.
 * Arguments     : device_type: for USB,  specify "usb".
 *
 * Return value  : media_port, media port that device is connected.
 ******************************************/
std::string query_device_status(std::string device_type)
{
    std::string media_port = "";
    /* Linux command to be executed */
    const char* command = "v4l2-ctl --list-devices";
    /* Open a pipe to the command and execute it */
    FILE* pipe = popen(command, "r");
    if (!pipe)
    {
        std::cerr << "[ERROR] Unable to open the pipe." << std::endl;
        return media_port;
    }
    /* Read the command output line by line */
    char buffer[128];
    size_t found;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        std::string response = std::string(buffer);
        found = response.find(device_type);
        if (found != std::string::npos)
        {
            fgets(buffer, sizeof(buffer), pipe);
            media_port = std::string(buffer);
            pclose(pipe);
            /* return media port*/
            return media_port;
        }
    }
    pclose(pipe);
    /* return media port*/
    return media_port;
}

/*****************************************
* Function Name : R_Kbhit_Thread
* Description   : Executes the Keyboard hit thread (checks if enter key is hit)
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Kbhit_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t kh_sem_check = 0;
    /*Variable to store the getchar() value*/
    int32_t c = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Key Hit Thread Starting\n");

    printf("************************************************\n");
    printf("* Press ENTER key to quit. *\n");
    printf("************************************************\n");

    /*Set Standard Input to Non Blocking*/
    errno = 0;
    ret = fcntl(0, F_SETFL, O_NONBLOCK);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to run fctnl(): errno=%d\n", errno);
        goto err;
    }

    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &kh_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != kh_sem_check)
        {
            goto key_hit_end;
        }

        c = getchar();
        if (EOF != c)
        {
            /* When key is pressed. */
            printf("key Detected.\n");
            goto err;
        }
        else
        {
            /* When nothing is pressed. */
            usleep(WAIT_TIME);
        }
    }

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto key_hit_end;

key_hit_end:
    printf("Key Hit Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Capture_Thread
* Description   : Executes the V4L2 capture with Capture thread.
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Capture_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t capture_sem_check = 0;
    int8_t ret = 0;

    printf("Capture Thread Starting\n");

    /* Capture stream of frames from camera using Gstreamer pipeline */
    cap.open(gstreamer_pipeline, CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Error opening video stream or camera !" << std::endl;
        return 0;
    }

    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &capture_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != capture_sem_check)
            {
                goto capture_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (capture_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        /* Capture USB camera image and stop updating the capture buffer */
        cap >> g_frame;
        if (g_frame.empty())
        {
            std::cout << "[INFO] Video ended or corrupted frame !\n";
            return 0;
        }

        if(flip_mode == 1)
        {
            cv::flip(g_frame, g_frame, 1);  //CPU
        }

        capture_start.store(0);
    } /*End of Loop*/

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    goto capture_end;

capture_end:
    printf("Capture Thread Terminated\n");
    pthread_exit(NULL);
}

void *R_Memcpy_Thread(void *threadid)
{
    int32_t ret = 0;
    int32_t memcpy_sem_check = 0;

    static int8_t memcpy_flag = 1;

    /*Image to be displayed on GUI*/
    cv::Mat input_image;
    cv::Mat padding_frame(IMAGE_WIDTH - IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

    printf("Memory copy Loop Starting\n");
    /*Memory copy Loop Start*/
    while(1)
    {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &memcpy_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != memcpy_sem_check)
            {
                goto memcpy_end;
            }

            if (!capture_start.load() && !img_processing_start.load() && !inference_start.load() && memcpy_flag == 1)
            {
                /* Copy captured image to inference buffer. This will be used in AI Inference Thread. */
                memcpy(inf_frame.data, g_frame.data, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);

                /*Image: CAM_IMAGE_WIDTH*CAM_IMAGE_HEIGHT (BGR) */
                input_image = inf_frame.clone();
                /*Add padding for keeping the aspect ratio: CAM_IMAGE_WIDTH*CAM_IMAGE_WIDTH (BGR) */
                cv::vconcat(input_image, padding_frame, input_image);

                /*Copy input data to drpai_buf for DRP-AI Pre-processing Runtime.*/
                memcpy( drpai_buf->mem, input_image.data, drpai_buf->size);
                /* Flush buffer */
                ret = buffer_flush_dmabuf(drpai_buf->idx, drpai_buf->size);
                if (0 != ret)
                {
                    goto err;
                }

                inference_start.store(1); /* Flag for AI Inference Thread. */
                img_processing_start.store(1);

                memcpy_flag = 0;
            }

            if (!img_processing_start.load() && !inference_start.load() && memcpy_flag == 0)
            {
                /* Copy captured image to inference buffer. This will be used in AI Inference Thread. */
                memcpy(img_frame.data, inf_frame.data, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
                capture_start.store(1); /* Flag for AI Inference Thread. */

                memcpy_flag = 1;
            }
        /*Wait for 1 TICK.*/
        usleep(WAIT_TIME);
    }/*End of Inference Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto memcpy_end;
/*Memory Copy Thread Termination*/
memcpy_end:
    /*To terminate the loop in AI Inference Thread.*/
    inference_start.store(0);

    printf("Memory Copy Thread Terminated\n");
    pthread_exit(NULL);
}

void *R_Inf_Thread(void *threadid)
{
    int32_t ret = 0;
    int32_t inf_sem_check = 0;

    printf("Inference Loop Starting\n");
    /*Inference Loop Start*/
    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != inf_sem_check)
            {
                goto ai_inf_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (inference_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        int ret = Head_Detection();
        if (ret != 0)
        {
            std::cerr << "[ERROR] Inference Not working !!! " << std::endl;
        }

        /* result copy */
        det_sub.resize(det.size());
        copy(det.begin(), det.end(), det_sub.begin());

        inference_start.store(0);
    }/*End of Inference Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
/*AI Thread Termination*/
ai_inf_end:
    /*To terminate the loop in Capture Thread.*/
    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}

int8_t R_Main_Process()
{
    /*Main Process Variables*/
    int8_t main_ret = 0;
    /*Semaphore Related*/
    int32_t main_sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    stringstream stream;
    string str = "";
    int32_t baseline = 10;
    uint8_t * img_buffer0;

    img_buffer0 = (unsigned char*) (malloc(DISP_OUTPUT_WIDTH*DISP_OUTPUT_HEIGHT*BGRA_CHANNEL));

    printf("Main Loop Starts\n");
    /*Main Loop Start*/
    while(1)
    {
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &main_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != main_sem_check)
            {
                goto main_proc_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (img_processing_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }

        /* Image processing time start */
        auto t14 = std::chrono::high_resolution_clock::now();

        cv::Mat output_image(DISP_OUTPUT_HEIGHT, DISP_OUTPUT_WIDTH , CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat pre_image(DISP_INF_HEIGHT, DISP_INF_WIDTH , CV_8UC3, cv::Scalar(0, 0, 0));

        /* Draw bounding box and Mosaic on the frame */
        draw_bounding_box();

        /*resize the image to the keep ratio size*/
        Size size(DISP_INF_WIDTH, DISP_INF_HEIGHT);
        resize(img_frame, pre_image, size);

        pre_image.copyTo(output_image(Rect(0, 60, DISP_INF_WIDTH, DISP_INF_HEIGHT)));

        /*Display frame */
        stream.str("");
        stream << "Head Count: " << HEAD_COUNT;
        str = stream.str();
        Size count_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - count_size.width - RIGHT_ALIGN_OFFSET), (HEAD_COUNT_STR_Y + count_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - count_size.width - RIGHT_ALIGN_OFFSET), (HEAD_COUNT_STR_Y + count_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_LARGE, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "AI Total Time: "<< fixed <<setprecision(1) << TOTAL_TIME<<" ms";
        str = stream.str();
        Size tot_time_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - tot_time_size.width - RIGHT_ALIGN_OFFSET), (T_TIME_STR_Y + tot_time_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_LARGE, Scalar(0, 255, 0), HC_CHAR_THICKNESS);
        stream.str("");
        stream << "Pre-Proc: "<< fixed <<setprecision(1) << PRE_PROC_TIME<<" ms";
        str = stream.str();
        Size pre_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - pre_proc_size.width - RIGHT_ALIGN_OFFSET), (PRE_TIME_STR_Y + pre_proc_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
        stream.str("");
        stream << "Inference: "<< fixed <<setprecision(1) << INF_TIME<<" ms";
        str = stream.str();
        Size inf_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - inf_size.width - RIGHT_ALIGN_OFFSET), (I_TIME_STR_Y + inf_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);
        stream.str("");
        stream << "Post-Proc: "<< fixed <<setprecision(1) << POST_PROC_TIME<<" ms";
        str = stream.str();
        Size post_proc_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_SMALL, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - post_proc_size.width - RIGHT_ALIGN_OFFSET), (P_TIME_STR_Y + post_proc_size.height)), FONT_HERSHEY_SIMPLEX,
                    CHAR_SCALE_SMALL, Scalar(255, 255, 255), HC_CHAR_THICKNESS);

        stream.str("");
        stream << "Image Proc: "<< fixed <<setprecision(1) << IMAGE_TIME<<" ms";
        str = stream.str();
        Size draw_size = getTextSize(str, FONT_HERSHEY_SIMPLEX,CHAR_SCALE_LARGE, HC_CHAR_THICKNESS, &baseline);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - draw_size.width - RIGHT_ALIGN_OFFSET), (370 + draw_size.height)), FONT_HERSHEY_SIMPLEX,
                CHAR_SCALE_LARGE, Scalar(0, 0, 0), 1.5*HC_CHAR_THICKNESS);
        putText(output_image, str,Point((DISP_OUTPUT_WIDTH - draw_size.width - RIGHT_ALIGN_OFFSET), (370 + draw_size.height)), FONT_HERSHEY_SIMPLEX,
                CHAR_SCALE_LARGE, Scalar(0, 255, 0), HC_CHAR_THICKNESS);

        namedWindow("Output Image", WND_PROP_FULLSCREEN);
        setWindowProperty("Output Image", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        string click_req = "Output Image";
        setMouseCallback(click_req,click_event,NULL);
        cv::Mat bgra_image;
        cv::cvtColor(output_image, bgra_image, cv::COLOR_BGR2BGRA);

        memcpy(img_buffer0, bgra_image.data, DISP_OUTPUT_WIDTH * DISP_OUTPUT_HEIGHT * BGRA_CHANNEL);
        wayland.commit(img_buffer0, NULL);

        img_processing_start.store(0);

        /* Image processing time end */
        auto t15 = std::chrono::high_resolution_clock::now();
        auto image_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(t15 - t14).count();
        IMAGE_TIME = image_processing_time/1000.0;
    }/*End of Loop*/

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_proc_end;
/*Main Processing Termination*/
main_proc_end:
    printf("Main Process Terminated\n");
    return main_ret;
}

int main(int argc, char *argv[])
{
    /*Multithreading Variables*/
    int32_t create_thread_ai  = -1;
    int32_t create_thread_key = -1;
    int32_t create_thread_capture = -1;
    int32_t create_thread_memcpy  = -1;
    int8_t ret_main = 0;
    int32_t ret = 0;
    int8_t main_proc = 0;
    int32_t sem_create = -1;
    std::string input_source = argv[1];
    std::cout << "Starting Head Count Top View Application" << std::endl;

    InOutDataType input_data_type;
    bool runtime_status = false;
    int drpai_fd;

    unsigned long OCA_list[16];
    /*Disable OpenCV Accelerator due to the use of multithreading */
    for (int i=0; i<16; i++)
    {
        OCA_list[i] = 0;    //disable
    }
    OCA_Activate( &OCA_list[0] );

    if (strcmp(argv[1],"USB") == 0)
    {
        if(argc == 2)
        {
            box_mode = 0;
            flip_mode = 0;
        }
        else if(argc == 3)
        {
            if(strcmp(argv[2],"BOX") == 0)
            {
                box_mode = 1;
                flip_mode = 0;
            }
            else if(strcmp(argv[2],"FLIP") == 0)
            {
                box_mode = 0;
                flip_mode = 1;
            }
            else
            {
                printf("Add the Box draw mode setting to the second argument of the execution command.\n");
                printf("Box draw mode is the setting to display bounding box on the screen.\n");
                printf("BOX : Display bounding box\n");
                printf("Execution command example:./head_count_topview_app USB BOX\n");
                printf("The above command displays bounding box.\n");
                return -1;
            }
        }
        else if(argc == 4)
        {
            if((strcmp(argv[2],"BOX") == 0 && strcmp(argv[3],"FLIP") == 0) || (strcmp(argv[3],"BOX") == 0 && strcmp(argv[2],"FLIP") == 0))
            {
                box_mode = 1;
                flip_mode = 1;
            }
            else
            {
                printf("Add the Box draw mode setting to the second argument of the execution command.\n");
                printf("Box draw mode is the setting to display bounding box on the screen.\n");
                printf("BOX : Display bounding box\n");
                printf("Execution command example:./head_count_topview_app USB BOX\n");
                printf("The above command displays bounding box.\n");
                return -1;
            }
        }
        else
        {
            printf("Add the Box draw mode setting to the second argument of the execution command.\n");
            printf("Box draw mode is the setting to display bounding box on the screen.\n");
            printf("BOX : Display bounding box\n");
            printf("Execution command example:./head_count_topview_app USB BOX\n");
            printf("The above command displays bounding box.\n");
            return -1;
        }
        drp_max_freq = DRP_MAX_FREQ;
        drp_freq = DRPAI_FREQ;
    }
    else
    {
        std::cout<<"Support for USB mode only."<<std::endl;
        return -1;
    }

    if (argc>5)
    {
        std::cerr << "[ERROR] Wrong number Arguments are passed " << std::endl;
        return 1;
    }

    errno = 0;
    drpai_fd = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd)
    {
        std::cerr << "[ERROR] Failed to open DRP-AI Driver : errno=" << errno << std::endl;
        return -1;
    }

    /*Load Label from label_list file*/
    label_file_map = load_label_file(label_list);
    if (label_file_map.empty())
    {
        fprintf(stderr,"[ERROR] Failed to load label file: %s\n", label_list.c_str());
        ret_main = -1;
        goto end_main;
    }

    /*Initialzie DRP-AI (Get DRP-AI memory address and set DRP-AI frequency)*/
    drpaimem_addr_start = init_drpai(drpai_fd);
    if (drpaimem_addr_start == 0)
    {
        close(drpai_fd);
        return -1;
    }

    /*Load pre_dir object to DRP-AI */
    ret = preruntime.Load(pre_dir);
    if (0 < ret)
    {
        fprintf(stderr, "[ERROR] Failed to run Pre-processing Runtime Load().\n");
        ret_main = -1;
        goto end_close_drpai;
    }

    /*Load model_dir structure and its weight to runtime object */
    runtime_status = runtime.LoadModel(model_dir, drpaimem_addr_start + DRPAI_MEM_OFFSET);

    if(!runtime_status)
    {
        std::cerr << "[ERROR] Failed to load model. " << std::endl;
        close(drpai_fd);
        return -1;
    }

    /*Get input data */
    input_data_type = runtime.GetInputDataType(0);
    if (InOutDataType::FLOAT32 == input_data_type)
    {
        /*Do nothing*/
    }
    else if (InOutDataType::FLOAT16 == input_data_type)
    {
        fprintf(stderr, "[ERROR] Input data type : FP16.\n");
        /*If your model input data type is FP16, use std::vector<uint16_t> for reading input data. */
        goto end_close_drpai;
    }
    else
    {
        fprintf(stderr, "[ERROR] Input data type : neither FP32 nor FP16.\n");
        goto end_close_drpai;
    }

    /*Initialize buffer for DRP-AI Pre-processing Runtime. */
    drpai_buf = (dma_buffer*)malloc(sizeof(dma_buffer));
    ret = buffer_alloc_dmabuf(drpai_buf, DRPAI_IN_WIDTH * DRPAI_IN_WIDTH * BGR_CHANNEL);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to Allocate DMA buffer for the drpai_buf\n");
        goto end_free_malloc;
    }

    std::cout << "[INFO] loaded runtime model :" << model_dir << "\n\n";
    switch (input_source_map[input_source])
    {
        /* Input Source : USB*/
        case 1:{
            std::cout << "[INFO] USB CAMERA \n";
            media_port = query_device_status("usb");
            gstreamer_pipeline = "v4l2src device=" + media_port + " ! video/x-raw, width=640, height=480 ! videoconvert ! appsink";

            /* Initialize waylad */
            ret = wayland.init(DISP_OUTPUT_WIDTH, DISP_OUTPUT_HEIGHT, BGRA_CHANNEL);
            if(0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
                ret_main = -1;
                goto end_close_drpai;
            }

            /*Termination Request Semaphore Initialization*/
            /*Initialized value at 1.*/
            sem_create = sem_init(&terminate_req_sem, 0, 1);
            if (0 != sem_create)
            {
                fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
                ret_main = -1;
                goto end_threads;
            }
            /*Create Key Hit Thread*/
            create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
            if (0 != create_thread_key)
            {
                fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
            /*Create Inference Thread*/
            create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
            if (0 != create_thread_ai)
            {
                sem_trywait(&terminate_req_sem);
                fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
            /*Create Capture Thread*/
            create_thread_capture = pthread_create(&capture_thread, NULL, R_Capture_Thread, NULL);
            if (0 != create_thread_capture)
            {
                sem_trywait(&terminate_req_sem);
                fprintf(stderr, "[ERROR] Failed to create Capture Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
            /*Create Memory Copy Thread*/
            create_thread_memcpy = pthread_create(&memcpy_thread, NULL, R_Memcpy_Thread, NULL);
            if (0 != create_thread_memcpy)
            {
                sem_trywait(&terminate_req_sem);
                fprintf(stderr, "[ERROR] Failed to create Memory Copy Thread.\n");
                ret_main = -1;
                goto end_threads;
            }
        }
        break;
    }

    /*Main Processing*/
    main_proc = R_Main_Process();
    if (0 != main_proc)
    {
        fprintf(stderr, "[ERROR] Error during Main Process\n");
        ret_main = -1;
    }
    goto end_threads;

end_threads:
    if(0 == create_thread_capture)
    {
        ret = wait_join(&capture_thread, CAPTURE_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Capture Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_memcpy)
    {
        ret = wait_join(&memcpy_thread, CAPTURE_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Memory Copy Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_ai)
    {
        ret = wait_join(&ai_inf_thread, AI_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit AI Inference Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_key)
    {
        ret = wait_join(&kbhit_thread, KEY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Key Hit Thread on time.\n");
            ret_main = -1;
        }
    }


    /*Delete Terminate Request Semaphore.*/
    if (0 == sem_create)
    {
        sem_destroy(&terminate_req_sem);
    }

    /* Exit the program */
    wayland.exit();

    goto end_close_drpai;

end_free_malloc:
    free(drpai_buf);
    drpai_buf = NULL;

    goto end_close_drpai;

end_close_drpai:
    /*Close DRP-AI Driver.*/
    if (0 < drpai_fd)
    {
        errno = 0;
        ret = close(drpai_fd);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to close DRP-AI Driver: errno=%d\n", errno);
            ret_main = -1;
        }
    }
    goto end_main;

end_main:
    printf("Application End\n");
    return ret_main;
}
