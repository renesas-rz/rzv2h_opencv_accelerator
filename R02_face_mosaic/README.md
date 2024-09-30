# Face Mosaic

## Application: Overview
This application is designed to detect and mosaic heads in the capture image of Camera and displays the result on HDMI screen.

This software could be useful in a variety of settings, such as retail stores, museums, and events.  
The AI model used for the sample application is [YOLOV3](https://arxiv.org/pdf/1804.02767.pdf).

### Supported Product
- RZ/V2H Evaluation Board Kit (RZ/V2H EVK)

### Input/Output
<table>
    <tr>
      <th>Board</th>
      <th>Input</th>
      <th>Output</th>
    </tr>
    <tr>
      <td>RZ/V2H EVK</td>
      <td>USB camera</td>
      <td>HDMI</td>
    </tr>
</table>


## Application: Requirements

### Hardware Requirements
<table>
    <tr>
      <th>For</th>
      <th>Equipment</th>
      <th>Details</th>
    </tr>
    <tr>
      <td rowspan="4">RZ/V2H</td>
      <td>RZ/V2H EVK</td>
      <td>Evaluation Board Kit for RZ/V2H.</td>
    </tr>
    <tr>
      <td>AC Adapter</td>
      <td>USB Power Delivery adapter for the board power supply.<br>
      100W is required.</td>
    </tr>
    <tr>
      <td>HDMI Cable</td>
      <td>Used to connect the HDMI Monitor and the board.<br>
      RZ/V2H EVK has HDMI port.</td>
    </tr>
    <tr>
      <td>USB Camera</td>
      <td>Used as a camera input source.</td>
    </tr>
    <tr>
      <td rowspan="8">Common</td>
      <td>USB Cable Type-C</td>
      <td>Connect AC adapter and the board.</td>
    </tr>
    <tr>
      <td>HDMI Monitor</td>
      <td>Used to display the graphics of the board.</td>
    </tr>
    <tr>
      <td>microSD card</td>
      <td>Used as the filesystem.<br>
      Must have over 16GB capacity of blank space.<br>
      Operating Environment: Transcend UHS-I microSD 300S 16GB</td>
    </tr>
    <tr>
      <td>Linux PC</td>
      <td>Used to build application and setup microSD card.<br>
      Operating Environment: Ubuntu 20.04</td>
    </tr>
    <tr>
      <td>SD card reader</td>
      <td>Used for setting up microSD card.<br></td>
    </tr>
    <tr>
      <td>USB Hub</td>
      <td>Used to connect USB Keyboard and USB Mouse to the board.</td>
    </tr>
    <tr>
      <td>USB Keyboard</td>
      <td>Used to type strings on the terminal of board.</td>
    </tr>
    <tr>
      <td>USB Mouse</td>
      <td>Used to operate the mouse on the screen of board.</td>
    </tr>
  </table>

>**Note:** All external devices will be attached to the board and does not require any driver installation (Plug n Play Type)

Connect the hardware as shown below.  

|RZ/V2H EVK |
|:---|
|<img src=./img/facemosaic_hw_conf_v2h.png width=600> |

>**Note 1:** When using the keyboard connected to RZ/V Evaluation Board, the keyboard layout and language are fixed to English.  
**Note 2:** For RZ/V2H EVK, there are USB 2.0 and USB 3.0 ports.  
USB camera needs to be connected to appropriate port based on its requirement.

## Application: Build Stage

>**Note:** User can skip to the [next stage (deploy)](#application-deploy-stage) if they do not want to build the application.  
All pre-built binaries are provided.

### Prerequisites
This section expects the user to have completed Step 5 of [Getting Started Guide](https://renesas-rz.github.io/rzv_ai_sdk/latest/getting_started.html) provided by Renesas. 

After completion of the guide, the user is expected of following things.
- AI SDK setup is done.
- Following docker container is running on the host machine.
    |Board | Docker container |
    |:---|:---|
    |RZ/V2H EVK|`rzv2h_ai_sdk_container`  |

    >**Note:** Docker environment is required for building the sample application. 


### Application File Generation
1. On your host machine, copy the repository from the GitHub to the desired location. 
    1. It is recommended to copy/clone the repository on the `data` folder, which is mounted on the Docker container. 
    ```sh
    cd <path_to_data_folder_on_host>/data
    git clone https://github.com/renesas-rz/rzv2h_opencv_accelerator/tree/apps/R02_face_mosaic
    ```
    >Note: This command will download the whole repository, which include all other applications.  
    If you have already downloaded the repository of the same version, you may not need to run this command.  

2. Run (or start) the docker container and open the bash terminal on the container.  
E.g., for RZ/V2H, use the `rzv2h_ai_sdk_container` as the name of container created from  `rzv2h_ai_sdk_image` docker image.  
    > Note that all the build steps/commands listed below are executed on the docker container bash terminal.  

3. Set your clone directory to the environment variable.  
    ```sh
    export PROJECT_PATH=/drp-ai_tvm/data
    ```
3. Go to the application source code directory.  
    ```sh
    cd ${PROJECT_PATH}/R02_face_mosaic/<SRC_DIR>
    ```
    |Board | `SRC_DIR` |
    |:---|:---|
    |RZ/V2H EVK|`src_v2h`  |

4. Create and move to the `build` directory.
    ```sh
    mkdir -p build && cd build
    ```
4. Build the application by following the commands below.  
    - For RZ/V2H,
      ```sh
      cmake -DCMAKE_TOOLCHAIN_FILE=./toolchain/runtime.cmake -DV2H=ON ..
      make -j$(nproc)
      ```
5. The following application file would be generated in the `${PROJECT_PATH}/R02_face_mosaic/<SRC_DIR>/build` directory
    - face_mosaic_app


## Application: Deploy Stage
### Prerequisites
This section expects the user to have completed Step 7-1 of [Getting Started Guide](https://renesas-rz.github.io/rzv_ai_sdk/latest/getting_started.html#step7) provided by Renesas. 

After completion of the guide, the user is expected of following things.
- microSD card setup is done.

### File Configuration
For the ease of deployment all the deployable files and folders are provided in following folders.  
|Board | `EXE_DIR` |
|:---|:---|
|RZ/V2H EVK|[exe_v2h](./exe_v2h)  |

Each folder contains following items.
|File | Details |
|:---|:---|
|licenses | License information of AI model. <br>Not necessary for running application. |
|face_mosaic_yolov3 | Model object files for deployment.<br>Pre-processing Runtime Object files included. |
|labels.txt | Label list for Object Detection. |
|face_mosaic_app | application file. |

### Instruction
1. Run following commands to download the necessary file.  
Replace each variable according to your board.  
    ```sh
    cd ${PROJECT_PATH}/R02_face_mosaic/<EXE_DIR>/face_mosaic_yolov3
    wget <URL>/<SO_FILE>
    ```
    |Board | `EXE_DIR` |`URL` |`SO_FILE` |File Location |
    |:---|:---|:---|:---|:---|
    |RZ/V2H EVK|[exe_v2h](./exe_v2h)  |<span style="font-size: small">`https://github.com/TakaakiSuezawa/test/releases/tag/apps_V1.00/`</span>  |<span style="font-size: small">`R02_face_mosaic_deploy_tvm_v2h-v100.so`</span> |[Release v1.00](https://github.com/TakaakiSuezawa/test/releases/tag/apps_V1.00/)  |

    - E.g., for RZ/V2H EVK, use following commands.
        ```sh
        cd ${PROJECT_PATH}/R02_face_mosaic/exe_v2h/face_mosaic_yolov3
        wget https://github.com/TakaakiSuezawa/test/releases/tag/apps_V1.00/R02_face_mosaic_deploy_tvm_v2h_v100.so
        ```
2. Rename the `R02_face_mosaic_deploy_*.so` to `deploy.so`.
    ```sh
    mv <SO_FILE> deploy.so
    ```
3. Copy the following files to the `/home/root/tvm` directory of the rootfs (SD Card) for the board.
    |File | Details |
    |:---|:---|
    |All files in `exe_v2h` directory | Including `deploy.so` file. |
    |`face_mosaic_app` application file | Generated the file according to [Application File Generation](#application-file-generation) |

4. Check if `libtvm_runtime.so` exists under `/usr/lib64` directory of the rootfs (SD card) on the board.

5. Folder structure in the rootfs (SD Card) would look like:
    ```
    |-- usr/
    |   `-- lib64/
    |       `-- libtvm_runtime.so
    `-- home/
        `-- root/
            `-- tvm/ 
                |-- face_mosaic_yolov3/
                |   |-- preprocess
                |   |-- deploy.json
                |   |-- deploy.params
                |   `-- deploy.so
                |-- labels.txt
                `-- face_mosaic_app
    ```
>**Note:** The directory name could be anything instead of `tvm`. If you copy the whole `EXE_DIR` folder on the board, you are not required to rename it `tvm`.

## Application: Run Stage

### Prerequisites
This section expects the user to have completed Step 7-3 of [Getting Started Guide](https://renesas-rz.github.io/rzv_ai_sdk/latest/getting_started.html#step7-3) provided by Renesas. 

After completion of the guide, the user is expected of following things.  
- The board setup is done.  
- The board is booted with microSD card, which contains the application file.  

### Instruction
1. On Board terminal, go to the `tvm` directory of the rootfs.
    ```sh
    cd /home/root/tvm
    ```

2. Run the application.
    ```sh
    ./face_mosaic_app USB
    ```
3. Following window shows up on HDMI screen.  

    |RZ/V2H EVK |
    |:---|
    |<img src=./img/facemosaic_v2h_normal.png width=350>  |

    On application window, following information is displayed.  
    - Camera capture  
    - Head Detection result (with mosaic)
    - Number of detected heads  
    - Processing time  
        - AI Total Time: Sum of all processing time below.  
        - Pre-Proc: Processing time taken for AI pre-processing. 
        - Inference: Processing time taken for AI inference.   
        - Post-Proc: Processing time taken for AI post-processing.<br>(excluding the time for drawing on HDMI screen).
        - Image Proc: Processing time taken for image processing.  

4. Following window shows up on HDMI screen in Bounding Box mode by ececuting the command below.
    ```sh
    ./face_mosaic_app USB BOX
    ```

    |RZ/V2H EVK |
    |:---|
    |<img src=./img/facemosaic_v2h_box.png width=350>  |

    Following window shows up on HDMI screen in Flip mode by ececuting the command below.
    ```sh
    ./face_mosaic_app USB FLIP
    ```

    |RZ/V2H EVK |
    |:---|
    |<img src=./img/facemosaic_v2h_flip.png width=350>  |
    >**Note:** Bounding Box mode and Flip mode can be used together.

5. To terminate the application, switch the application window to the terminal by using `Super(windows key)+Tab` and press ENTER key on the terminal of the board.


## Application: Configuration 
### AI Model
- YOLOv3: [Darknet](https://pjreddie.com/darknet/yolo/)  
- Dataset: *[HollywoodHeads](https://www.di.ens.fr/willow/research/headdetection/) *[Head_data](https://www.kaggle.com/datasets/houssad/head-data) *[RGBD_Indoor_Dataset](https://drive.google.com/file/d/1fOub9LcNqfDlr-mEcdnenAJWw-rqWGmG/view)

Input size: 1x3x416x416  
Output1 size: 1x13x13x18  
Output2 size: 1x26x26x18  
Output3 size: 1x52x52x18   

### AI inference time
|Board | AI inference time|
|:---|:---|
|RZ/V2H EVK | Approximately 25ms  |

### Processing

|Processing | Details |
|:---|:---|
|Pre-processing | Processed by DRP-AI. <br> |
|Inference | Processed by DRP-AI and CPU. |
|Post-processing | Processed by CPU. |


### Image buffer size

|Board | Camera capture buffer size|HDMI output buffer size|
|:---|:---|:---|
|RZ/V2H EVK | VGA (640x480) in BGR format  | FHD (1920x1080) in BGRA format  |
  
> **Note:** For RZ/V2H, this application allocates the DRP-AI input buffer with **640x640** resolution in order to maintain the same aspect ratio with **416x416** square size of YOLOv3 input shape after the resize pre-processing.  
  
Following is the buffer flow for RZ/V2H application.  

<img src=./img/facemosaic_buffer_flow.png width="800">


## Reference
- For RZ/V2H EVK, this application supports USB camera only with 640x480 resolution.  
FHD resolution is supported by e-CAM22_CURZH camera (MIPI).  
Please refer to following URL for how to change camera input to MIPI camera.  
[https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications](https://renesas-rz.github.io/rzv_ai_sdk/latest/about-applications#mipi).  

## License
For AI model, see following directory..  
|Board | AI Model | License directory|
|:---|:---|:---|
|RZ/V2H EVK | YOLOv3  | `exe_v2h/licenses`  |
