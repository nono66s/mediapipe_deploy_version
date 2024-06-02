import os 
import cv2
import math
import json
import shutil
import datetime
import xlsxwriter 

import torch             #torch.nn , contains all of the Pytorch building block for Neural Network
import torchvision
import torchsummary
import torchinfo
import sys


import numpy as np
import mediapipe as mp

from utils.Config import config
from utils.Groundtruth_Json_Maker import Dir2JSON

def euclidean_distance(point1, point2):
    """
    計算兩點之間的歐式距離

    參數:
    point1: tuple, 第一個點的座標 (x1, y1, z1, ...)
    point2: tuple, 第二個點的座標 (x2, y2, z2, ...)

    返回值:
    float, 兩點之間的歐式距離
    """
    distance = math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    return distance

def space_angel(angel_point,a_point,b_point):

    a_vector = a_point-angel_point
    b_vector = b_point-angel_point

    a_length = ((a_point[0]-angel_point[0])**2+(a_point[1]-angel_point[1])**2+(a_point[2]-angel_point[2])**2)**0.5
    b_length = ((b_point[0]-angel_point[0])**2+(b_point[1]-angel_point[1])**2+(b_point[2]-angel_point[2])**2)**0.5

    angel_radians=np.arccos(np.dot(a_vector,b_vector)/(a_length*b_length))

    return math.degrees(angel_radians)

def space_length(a_point,b_point):

    return ((a_point[0]-b_point[0])**2+(a_point[1]-b_point[1])**2+(a_point[2]-b_point[2])**2)**0.5

def angle_with_xy_plane(point1, point2):
    """
    计算空间中两个点形成的向量与 xy 平面的夹角。

    参数：
        point1: 第一个点的坐标，格式为 [x, y, z]。
        point2: 第二个点的坐标，格式为 [x, y, z]。

    返回值：
        angle_with_xy_plane: 向量与 xy 平面的夹角（单位：度）。
    """
    # 计算由两个点形成的向量
    vector = np.array(point2) - np.array(point1)

    # 计算向量与 xy 平面法线向量的夹角
    normal_to_xy_plane = np.array([0, 0, 1])  # xy 平面的法线向量
    dot_product = np.dot(vector, normal_to_xy_plane)
    norm_vector = np.linalg.norm(vector)
    norm_normal = np.linalg.norm(normal_to_xy_plane)
    cosine_angle = dot_product / (norm_vector * norm_normal)

    # 将余弦值转换为角度
    angle_in_radians = np.arccos(cosine_angle)
    angle_in_degrees = np.degrees(angle_in_radians)

    # 取余角
    angle_with_xy_plane = 90 - angle_in_degrees

    return angle_with_xy_plane

#---------------------------------------------------------#
#   影片準備
#---------------------------------------------------------#

def base_parameter (video_in_dir) :

    video_name  = str(video_in_dir.split("\\")[-1])   # 影像名稱   
    video_name  = str(video_name.split(".")[0])   # 影像名稱   
    print(f"current video {video_name}")

    video_label = str(video_in_dir.split("\\")[1])   # 影像名稱   

    cap = cv2.VideoCapture(video_in_dir)  # 影像路徑，輸入成影像物件

    frame  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 影像總禎數
    fps    = int(cap.get(cv2.CAP_PROP_FPS))          # 影像FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影像寬度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 影像高度
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')         # 設定影片的解碼器為 MJPG

    #---------------------------------------------------------#
    #   輸出位置            :output 
    #   創建單次輸出資料夾   :output_2023_03_08_07_57_14
    #---------------------------------------------------------#

    time_str     = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    output_dir   = os.path.join(config.preprocessing_path, "output_" +video_name+"_" + str(time_str))        

    os.makedirs(output_dir)

    shutil.copy(video_in_dir,output_dir)

    #---------------------------------------------------------#
    #   影片物件
    #---------------------------------------------------------#

    out = cv2.VideoWriter(os.path.join(output_dir,'output.mp4'), fourcc, fps, (width,height))  # 產生空的影片

    #---------------------------------------------------------#
    #   建立儲存用的json檔案
    #---------------------------------------------------------#

    now = datetime.datetime.now()               # 檔案建立的時間

    mediapipe_json_log = dict(
                                origin_video_name               = video_name,
                                video_label                     = video_label,
                                year                            = now.year,
                                date_created                    = now.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                date_end                        = [],
                                duration                        = [],
                                fps                             = fps,
                                width                           = width,
                                height                          = height,
                                body_landmark_name              = config.body_landmark_name,
                                unhealthy_frame                 = [],   

                                left_hand_angel                 = [],
                                right_hand_angel                = [],
                                left_foot_angel                 = [],
                                right_foot_angel                = [],
                                    
                                left_body_length                = [],
                                right_body_length               = [],

                                detection                       = [],

            )

    #---------------------------------------------------------#
    #   medipipe縮寫
    #---------------------------------------------------------#

    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式

    #---------------------------------------------------------#
    #   啟用姿勢偵測
    #---------------------------------------------------------#
    mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
    pose    = mp_pose.Pose(
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) 

    frame_count=0

    while cap.isOpened():
        ret, img = cap.read()

        frame_count+=1
        frame_templet=dict(
                            Current_frame      = frame_count,
                            Current_second     = math.floor((frame_count)/fps),

                            frame_state        = None,

                            left_hand_angel    = None,
                            right_hand_angel   = None,
                            left_foot_angel    = None,
                            right_foot_angel   = None,

                            face_angle         = None,
                            shoulder_angle     = None,
                            hip_angle          = None,

                            left_body_length   = None,
                            right_body_length  = None,

                            landmark_list      = []
                        )

        if ret:
            
            #img = cv2.resize(img,(520,300))              # 縮小尺寸，加快演算速度
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = pose.process(img2)                  # 取得姿勢偵測結果
            # 根據姿勢偵測結果，標記身體節點和骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            for i in range(33):
                
                # 要注意添加新的點連空格都要一樣
                if list(config.body_landmark_name)[i] in ["LEFT_HIP ","RIGHT_HIP","LEFT_KNEE ","RIGHT_KNEE ","LEFT_ANKLE","RIGHT_ANKLE ",
                                                           "LEFT_SHOULDER ","RIGHT_SHOULDER ","LEFT_ELBOW ","RIGHT_ELBOW ","LEFT_WRIST ","RIGHT_WRIST ",
                                                           'LEFT_EYE','RIGHT_EYE ','NOSE',]:
                    try:
                        frame_templet['landmark_list'].append(dict(
                                                                        landmark_id             = i,
                                                                        landmark_name           = list(config.body_landmark_name)[i],
                                                                        x                       = results.pose_landmarks.landmark[i].x,           # normalized width
                                                                        y                       = results.pose_landmarks.landmark[i].y,           # normalized height
                                                                        z                       = results.pose_landmarks.landmark[i].z,           # normalized landmark depth (smaller the value the closer the landmark is to the camera)
                                                                        visibility              = results.pose_landmarks.landmark[i].visibility,  # the point is visible or hidden
                                                                        
                                                                        velocity                = None,
                                                                        accelerate              = None,

                                                                        normallized_velocity    = None,

                                                            ))
                    except:
                        frame_templet["frame_state"]="broken"

                        if int(frame_count) not in mediapipe_json_log['unhealthy_frame']:
                            mediapipe_json_log['unhealthy_frame'].append(int(frame_count))
                        pass

            if frame_templet["frame_state"] == None:          # 代表正常
                
                landmark_list=frame_templet["landmark_list"]

                #print(landmark_list)

                LEFT_HIP       = np.array([landmark_list[0]['x'],landmark_list[0]['y'],landmark_list[0]['z']])
                RIGHT_HIP      = np.array([landmark_list[1]['x'],landmark_list[1]['y'],landmark_list[1]['z']])
                LEFT_KNEE      = np.array([landmark_list[2]['x'],landmark_list[2]['y'],landmark_list[2]['z']])
                
                RIGHT_KNEE     = np.array([landmark_list[3]['x'],landmark_list[3]['y'],landmark_list[3]['z']])
                LEFT_ANKLE     = np.array([landmark_list[4]['x'],landmark_list[4]['y'],landmark_list[4]['z']])
                RIGHT_ANKLE    = np.array([landmark_list[5]['x'],landmark_list[5]['y'],landmark_list[5]['z']])

                LEFT_SHOULDER  = np.array([landmark_list[6]['x'],landmark_list[6]['y'],landmark_list[6]['z']]) 
                RIGHT_SHOULDER = np.array([landmark_list[7]['x'],landmark_list[7]['y'],landmark_list[7]['z']])
                LEFT_ELBOW     = np.array([landmark_list[8]['x'],landmark_list[8]['y'],landmark_list[8]['z']]) 

                RIGHT_ELBOW    = np.array([landmark_list[9]['x'],landmark_list[9]['y'],landmark_list[9]['z']])
                LEFT_WRIST     = np.array([landmark_list[10]['x'],landmark_list[10]['y'],landmark_list[10]['z']])
                RIGHT_WRIST    = np.array([landmark_list[11]['x'],landmark_list[11]['y'],landmark_list[11]['z']])

                LEFT_EYE       = np.array([landmark_list[11]['x'],landmark_list[11]['y'],landmark_list[11]['z']])
                RIGHT_EYE      = np.array([landmark_list[12]['x'],landmark_list[12]['y'],landmark_list[12]['z']])
                NOSE           = np.array([landmark_list[13]['x'],landmark_list[13]['y'],landmark_list[13]['z']])

                
                frame_templet['left_hand_angel']  = space_angel(LEFT_ELBOW,LEFT_SHOULDER,LEFT_WRIST)
                frame_templet['right_hand_angel'] = space_angel(RIGHT_ELBOW,RIGHT_SHOULDER,RIGHT_WRIST)

                frame_templet['left_foot_angel']  = space_angel(LEFT_KNEE,LEFT_HIP,LEFT_ANKLE)
                frame_templet['right_foot_angel'] = space_angel(RIGHT_KNEE,RIGHT_HIP,RIGHT_ANKLE)

                frame_templet['left_body_length']  = space_length(LEFT_SHOULDER,LEFT_HIP)
                frame_templet['right_body_length'] = space_length(RIGHT_SHOULDER,RIGHT_HIP)

                frame_templet['face_angle']       = angle_with_xy_plane(LEFT_EYE,RIGHT_EYE)
                frame_templet['shoulder_angle']   = angle_with_xy_plane(LEFT_SHOULDER,RIGHT_SHOULDER)
                frame_templet['hip_angle']        = angle_with_xy_plane(LEFT_HIP,RIGHT_HIP)

                mediapipe_json_log['left_hand_angel'].append(frame_templet['left_hand_angel'])
                mediapipe_json_log['right_hand_angel'].append(frame_templet['right_hand_angel'])

                mediapipe_json_log['left_foot_angel'].append(frame_templet['left_foot_angel'])
                mediapipe_json_log['right_foot_angel'].append(frame_templet['right_foot_angel'])
    
                mediapipe_json_log['left_body_length'].append(frame_templet['left_body_length'])
                mediapipe_json_log['right_body_length'].append(frame_templet['right_body_length'])


            mediapipe_json_log['detection'].append(frame_templet)

            cv2.imshow('nono_studio', img)

            out.write(img)

        elif not ret:
            frame_templet["frame_state"]="empty"
            mediapipe_json_log['unhealthy_frame'].append(int(frame_count))
            mediapipe_json_log['detection'].append(frame_templet)

        if cv2.waitKey(1) == ord('q'):    
            break                       # 按下 q 鍵停止

        if sum((np.array(mediapipe_json_log['unhealthy_frame'])>4500)==True)>5:  # 大於  4500時才管 (3分鐘)

            cap.release()
            out.release
            cv2.destroyAllWindows()

    end=datetime.datetime.now()
    mediapipe_json_log['date_end'] = end.strftime('%Y-%m-%d %H:%M:%S.%f')
    mediapipe_json_log['duration'] = str(end-now)

    with open(os.path.join(output_dir,video_name.split('.')[0]+'_frame.json'), 'a') as f:
        json.dump(
            mediapipe_json_log, 
            f, 
            indent= 4
            )
        
    return os.path.join(output_dir,video_name.split('.')[0]+'_frame.json')







def calculate_velocity(i):

    with open (i,"r") as f:
        datas=json.load(f)

        LEFT_BODY_LENGTH  = np.mean(np.array(datas["left_body_length"]))
        RIGHT_BODY_LENGTH = np.mean(np.array(datas["right_body_length"]))

        BODY_LENGTH=np.mean(LEFT_BODY_LENGTH+RIGHT_BODY_LENGTH)

        for j in datas["detection"]: 
            if j["frame_state"]==None: # 健康的frame
                # j                                      當前
                # datas["detection"][j["Current_frame"]] 下一筆資料

                nex = datas["detection"][j["Current_frame"]]

                if nex["frame_state"] == None: # 如果下一筆資料是健康的frame
                    for land_num, point in enumerate(nex["landmark_list"]):

                        land_data = j["landmark_list"][land_num]
                        

                        point_cur = (land_data["x"], land_data["y"], land_data["z"])
                        point_nex = (point["x"], point["y"], point["z"])
                        distance = euclidean_distance(point_cur, point_nex)
                        velocity_of_next = distance/(1/datas["fps"])

                        # 經Normalize後像素點位置/FPS =所得速度
                        datas["detection"][j["Current_frame"]]["landmark_list"][land_num]["velocity"]=velocity_of_next
                        print(velocity_of_next)

                        # 經Normalize後像素點位置/FPS =所得速度
                        datas["detection"][j["Current_frame"]]["landmark_list"][land_num]["normallized_velocity"]=velocity_of_next/BODY_LENGTH

        #　E:\Deep_Learning\mediapipe\dataset\Medipipe處理後\WM異常\
        # 　output_2023_12_08_12_38_05\WM_A01王晅F233739096_20180206_45w5d_frame.json

        with open(os.path.join(ｉ.rsplit("_", maxsplit=1)[0]+'_add_vel.json'), 'a') as nf:
            json.dump(
                datas, 
                nf, 
                indent= 4
                )
            
        return os.path.join(ｉ.rsplit("_", maxsplit=1)[0]+'_add_vel.json')


def calculate_acc(i):
    
    with open (i,"r") as f:
        datas=json.load(f)

        for j in datas["detection"]: 
            if j["frame_state"]==None : # 健康的frame
                if j["landmark_list"][1]["velocity"]:
            
                    # j                                      當前
                    # datas["detection"][j["Current_frame"]] 下一筆資料

                    nex = datas["detection"][j["Current_frame"]]

                    if nex["frame_state"] == None : # 如果下一筆資料是健康的frame
                        if nex["landmark_list"][1]["velocity"]:

                            for land_num, point in enumerate(nex["landmark_list"]):

                                land_data = j["landmark_list"][land_num]
                                
                                point_cur_vel = land_data["velocity"]
                                point_nex_vel = point["velocity"]

                                point_cur_norm_vel = land_data["normallized_velocity"]
                                point_nex_norm_vel = point["normallized_velocity"]

                                acc_of_next      = (point_nex_vel-point_cur_vel)/((nex["Current_frame"]-j["Current_frame"])/datas["fps"])
                                norm_acc_of_next = (point_nex_norm_vel-point_cur_norm_vel)/((nex["Current_frame"]-j["Current_frame"])/datas["fps"])

                                # 經Normalize後像素點位置/FPS =所得速度
                                datas["detection"][j["Current_frame"]]["landmark_list"][land_num]["accelerate"]=acc_of_next
                                print(acc_of_next)

                                # 經Normalize後像素點位置/FPS =所得速度
                                datas["detection"][j["Current_frame"]]["landmark_list"][land_num]["normallized_accelerate"]=norm_acc_of_next

        #　E:\Deep_Learning\mediapipe\dataset\Medipipe處理後\WM異常\
        # 　output_2023_12_08_12_38_05\WM_A01王晅F233739096_20180206_45w5d_frame.json

        with open(os.path.join(ｉ.rsplit("_", maxsplit=1)[0]+'_add_vel_acc.json'), 'a') as nf:
            json.dump(
                datas, 
                nf, 
                indent= 4
                )
            
        return os.path.join(ｉ.rsplit("_", maxsplit=1)[0]+'_add_vel_acc.json')

def write_xlsx(i):
    
    print(i)
    name = os.path.basename(i)
    name=name.rsplit(".", maxsplit=1)[0]

    workbook = xlsxwriter.Workbook(os.path.join("output",name+'_excel.xlsx'))
    worksheet = workbook.add_worksheet()

    title = [   
                "Current_frame",
                "Current_second",
                "frame_state",
                "vel_state",
                
                "left_hand_angel","right_hand_angel",
                "left_foot_angel","right_foot_angel",

                "face_angle","shoulder_angle","hip_angle",
                "left_body_length","right_body_length",

                "NOSE","LEFT_EYE","RIGHT_EYE ",
                "LEFT_SHOULDER ","RIGHT_SHOULDER ","LEFT_ELBOW ","RIGHT_ELBOW ","LEFT_WRIST ","RIGHT_WRIST ",
                "LEFT_HIP ","RIGHT_HIP","LEFT_KNEE ","RIGHT_KNEE ","LEFT_ANKLE","RIGHT_ANKLE ",
                
                "NOSE_normallized_velocity","LEFT_EYE_normallized_velocity","RIGHT_EYE_normallized_velocity",
                "LEFT_SHOULDER_normallized_velocity ","RIGHT_SHOULDER_normallized_velocity ","LEFT_ELBOW_normallized_velocity ","RIGHT_ELBOW_normallized_velocity ","LEFT_WRIST_normallized_velocity ","RIGHT_WRIST_normallized_velocity ",
                "LEFT_HIP_normallized_velocity ","RIGHT_HIP_normallized_velocity","LEFT_KNEE_normallized_velocity ","RIGHT_KNEE_normallized_velocity ","LEFT_ANKLE_normallized_velocity","RIGHT_ANKLE_normallized_velocity ",

                "NOSE_x ","LEFT_EYE_x ","RIGHT_EYE_x ",
                "LEFT_SHOULDER_x ","RIGHT_SHOULDER_x ","LEFT_ELBOW_x ","RIGHT_ELBOW_x ","LEFT_WRIST_x ","RIGHT_WRIST_x ",
                "LEFT_HIP_x ","RIGHT_HIP_x ","LEFT_KNEE_x ","RIGHT_KNEE_x ","LEFT_ANKLE_x ","RIGHT_ANKLE_x ",

                "NOSE_y ","LEFT_EYE_y ","RIGHT_EYE_y ",
                "LEFT_SHOULDER_y ","RIGHT_SHOULDER_y ","LEFT_ELBOW_y ","RIGHT_ELBOW_y ","LEFT_WRIST_y ","RIGHT_WRIST_y ",
                "LEFT_HIP_y ","RIGHT_HIP_y ","LEFT_KNEE_y ","RIGHT_KNEE_y ","LEFT_ANKLE_y ","RIGHT_ANKLE_y ",

                "NOSE_z ","LEFT_EYE_z ","RIGHT_EYE_z ",
                "LEFT_SHOULDER_z ","RIGHT_SHOULDER_z ","LEFT_ELBOW_z ","RIGHT_ELBOW_z ","LEFT_WRIST_z ","RIGHT_WRIST_z ",
                "LEFT_HIP_z ","RIGHT_HIP_z ","LEFT_KNEE_z ","RIGHT_KNEE_z ","LEFT_ANKLE_z ","RIGHT_ANKLE_z ",


                "NOSE_accelerate","LEFT_EYE_accelerate","RIGHT_EYE_accelerate",
                "LEFT_SHOULDER_accelerate ","RIGHT_SHOULDER_accelerate ","LEFT_ELBOW_accelerate ","RIGHT_ELBOW_accelerate ","LEFT_WRIST_accelerate ","RIGHT_WRIST_accelerate ",
                "LEFT_HIP_accelerate ","RIGHT_HIP_accelerate","LEFT_KNEE_accelerate ","RIGHT_KNEE_accelerate ","LEFT_ANKLE_accelerate","RIGHT_ANKLE_accelerate ",

                "NOSE_normallized_accelerate","LEFT_EYE_normallized_accelerate","RIGHT_EYE_normallized_accelerate",
                "LEFT_SHOULDER_normallized_accelerate ","RIGHT_SHOULDER_normallized_accelerate ","LEFT_ELBOW_normallized_accelerate ","RIGHT_ELBOW_normallized_accelerate ","LEFT_WRIST_normallized_accelerate ","RIGHT_WRIST_normallized_accelerate ",
                "LEFT_HIP_normallized_accelerate ","RIGHT_HIP_normallized_accelerate","LEFT_KNEE_normallized_accelerate ","RIGHT_KNEE_normallized_accelerate ","LEFT_ANKLE_normallized_accelerate","RIGHT_ANKLE_normallized_accelerate ",


            ]


    for j,v in enumerate(title) :
        worksheet.write(0, j, j+1)
        worksheet.write(1, j, v)

    row = 2   

    with open (i,"r") as f:
        datas=json.load(f)
        tmp=[]
        for k in datas["detection"]:
            if k["frame_state"]==None and k["landmark_list"][0]["velocity"] != None and "normallized_accelerate" in k["landmark_list"][0]: # 健康的frame
                print(k["landmark_list"][0])
                # 因為使用五秒的 中間會有空白 所以隨意抓一個點判斷

                tmp=[
                        k["Current_frame"],k["Current_second"],1,1,

                        k["left_hand_angel"],k["right_hand_angel"],
                        k["left_foot_angel"],k["right_foot_angel"],

                        k["face_angle"],k["shoulder_angle"],k["hip_angle"],
                        k["left_body_length"],k["right_body_length"],

                        k["landmark_list"][0]["velocity"],k["landmark_list"][1]["velocity"],
                        k["landmark_list"][2]["velocity"],k["landmark_list"][3]["velocity"],
                        k["landmark_list"][4]["velocity"],k["landmark_list"][5]["velocity"],
                        k["landmark_list"][6]["velocity"],k["landmark_list"][7]["velocity"],
                        k["landmark_list"][8]["velocity"],k["landmark_list"][9]["velocity"],
                        k["landmark_list"][10]["velocity"],k["landmark_list"][11]["velocity"],
                        k["landmark_list"][12]["velocity"],k["landmark_list"][13]["velocity"],
                        k["landmark_list"][14]["velocity"],
                        
                        k["landmark_list"][0]["normallized_velocity"],k["landmark_list"][1]["normallized_velocity"],
                        k["landmark_list"][2]["normallized_velocity"],k["landmark_list"][3]["normallized_velocity"],
                        k["landmark_list"][4]["normallized_velocity"],k["landmark_list"][5]["normallized_velocity"],
                        k["landmark_list"][6]["normallized_velocity"],k["landmark_list"][7]["normallized_velocity"],
                        k["landmark_list"][8]["normallized_velocity"],k["landmark_list"][9]["normallized_velocity"],
                        k["landmark_list"][10]["normallized_velocity"],k["landmark_list"][11]["normallized_velocity"],
                        k["landmark_list"][12]["normallized_velocity"],k["landmark_list"][13]["normallized_velocity"],
                        k["landmark_list"][14]["normallized_velocity"],

                        k["landmark_list"][0]["x"],k["landmark_list"][1]["x"],
                        k["landmark_list"][2]["x"],k["landmark_list"][3]["x"],
                        k["landmark_list"][4]["x"],k["landmark_list"][5]["x"],
                        k["landmark_list"][6]["x"],k["landmark_list"][7]["x"],
                        k["landmark_list"][8]["x"],k["landmark_list"][9]["x"],
                        k["landmark_list"][10]["x"],k["landmark_list"][11]["x"],
                        k["landmark_list"][12]["x"],k["landmark_list"][13]["x"],
                        k["landmark_list"][14]["x"],

                        k["landmark_list"][0]["y"],k["landmark_list"][1]["y"],
                        k["landmark_list"][2]["y"],k["landmark_list"][3]["y"],
                        k["landmark_list"][4]["y"],k["landmark_list"][5]["y"],
                        k["landmark_list"][6]["y"],k["landmark_list"][7]["y"],
                        k["landmark_list"][8]["y"],k["landmark_list"][9]["y"],
                        k["landmark_list"][10]["y"],k["landmark_list"][11]["y"],
                        k["landmark_list"][12]["y"],k["landmark_list"][13]["y"],
                        k["landmark_list"][14]["y"],

                        k["landmark_list"][0]["z"],k["landmark_list"][1]["z"],
                        k["landmark_list"][2]["z"],k["landmark_list"][3]["z"],
                        k["landmark_list"][4]["z"],k["landmark_list"][5]["z"],
                        k["landmark_list"][6]["z"],k["landmark_list"][7]["z"],
                        k["landmark_list"][8]["z"],k["landmark_list"][9]["z"],
                        k["landmark_list"][10]["z"],k["landmark_list"][11]["z"],
                        k["landmark_list"][12]["z"],k["landmark_list"][13]["z"],
                        k["landmark_list"][14]["z"],

                        k["landmark_list"][0]["accelerate"],k["landmark_list"][1]["accelerate"],
                        k["landmark_list"][2]["accelerate"],k["landmark_list"][3]["accelerate"],
                        k["landmark_list"][4]["accelerate"],k["landmark_list"][5]["accelerate"],
                        k["landmark_list"][6]["accelerate"],k["landmark_list"][7]["accelerate"],
                        k["landmark_list"][8]["accelerate"],k["landmark_list"][9]["accelerate"],
                        k["landmark_list"][10]["accelerate"],k["landmark_list"][11]["accelerate"],
                        k["landmark_list"][12]["accelerate"],k["landmark_list"][13]["accelerate"],
                        k["landmark_list"][14]["accelerate"],

                        k["landmark_list"][0]["normallized_accelerate"],k["landmark_list"][1]["normallized_accelerate"],
                        k["landmark_list"][2]["normallized_accelerate"],k["landmark_list"][3]["normallized_accelerate"],
                        k["landmark_list"][4]["normallized_accelerate"],k["landmark_list"][5]["normallized_accelerate"],
                        k["landmark_list"][6]["normallized_accelerate"],k["landmark_list"][7]["normallized_accelerate"],
                        k["landmark_list"][8]["normallized_accelerate"],k["landmark_list"][9]["normallized_accelerate"],
                        k["landmark_list"][10]["normallized_accelerate"],k["landmark_list"][11]["normallized_accelerate"],
                        k["landmark_list"][12]["normallized_accelerate"],k["landmark_list"][13]["normallized_accelerate"],
                        k["landmark_list"][14]["normallized_accelerate"],
                    ]
                

                for index,value in enumerate(tmp) :
                    worksheet.write(row, index, value)
                row+=1

    workbook.close()

def dataset_getitem(input_path):
    
    with open(input_path, 'r', encoding="utf-8") as f:   
        
        dataset=json.load(f)

        # datasets_contain =   [],
        # categories       =   {},
        # datas_name       =   [],
        # datas            =   {},
        # ground_truth     =   {},

        # "categories": {
        #     "Abnormal_CS": "E:\\Deep_Learning\\AlexNet_pytorch\\model_data\\baby_datasets\\Abnormal_CS",
        #     "Abnormal_PR": "E:\\Deep_Learning\\AlexNet_pytorch\\model_data\\baby_datasets\\Abnormal_PR",
        #     "Normal": "E:\\Deep_Learning\\AlexNet_pytorch\\model_data\\baby_datasets\\Normal"
        # },

        sequence  = torch.tensor(dataset["datas"][str(input_path)], dtype=torch.float32)   # transform img to tensor
        
        # "output\\WM_N02王臣睿A133595391＿20180710＿44w1d_add_add_vel_acc_input.json"

        # print(sequence.shape) # (113, 2997)
        # (batch_size, seq_length, input_size)
        sequence  = sequence.permute(1,0)                                       # (input_size, seq_length) -> (seq_length, input_size) 

    return sequence

def model():

    CURRENT_MODEL_NAME= ['VGG16','LSTM'][-1]
    
    if CURRENT_MODEL_NAME=='LSTM':

        from model_architecture.LSTM import LSTM

        config.seq_length,config.input_size = 1500, 36

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    #--------------------------------------------------------------------------------------------------------------#
    # 解決衝突問題
    # 預設不允許同時重續訪問同一個庫
    # https://stackoverflow.com/questions/65734044/kernel-appears-to-have-died-jupyter-notebook-python-matplotlib 
    #--------------------------------------------------------------------------------------------------------------#

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    #----------------------------------------------------------------#
    # 確認所使用版本
    #----------------------------------------------------------------#
    # check python version and info
    # https://phoenixnap.com/kb/check-tensorflow-version code reference
    #----------------------------------------------------------------#

    version={
        "torch"         :torch.__version__,
        "torchvision"   :torchvision.__version__,
        "Python"        :sys.version,
        "Version info"  :str(sys.version_info),
    }

    for i,(k,v) in enumerate(version.items()):
        print(f"#{i:<3} {k:15} version is {v:10} ")

    #----------------------------------------------------------------#
    # 確認cuda是否正常開啟
    #----------------------------------------------------------------#
    # 若正常開啟則選擇GPU用於訓練
    #----------------------------------------------------------------#

    print('cuda is_available :', torch.cuda.is_available())
    print('cuda device count :', torch.cuda.device_count())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")


    #----------------------------------------------------------------#
    # -> ['Abnormal_CS', 'Abnormal_PR', 'Normal']
    #----------------------------------------------------------------#

    label_name_list = ['Abnormal_CS', 'Abnormal_PR', 'Normal']

    #----------------------------------------------------------------#
    # 轉換為帶有編號的字典
    #----------------------------------------------------------------#
    # -> {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    #----------------------------------------------------------------#
    # -> {'Abnormal_CS': 0, 'Abnormal_PR': 1, 'Normal': 2} 
    #----------------------------------------------------------------#

    label_map = {'Abnormal_CS': 0, 'Abnormal_PR': 1, 'Normal': 2} 


    model = LSTM(label_name_list,input_size=config.input_size).to(device)



    model_save_folder=f'./model_output/{CURRENT_MODEL_NAME}/'
    model_weight_folder=os.path.join(model_save_folder ,os.listdir(f'./model_output/{CURRENT_MODEL_NAME}')[-1])
    lastest_weight = os.path.join(model_weight_folder+"/model_weights",os.listdir(model_weight_folder+"/model_weights")[-1])

    load_pretrain_weights=config.specific_weights_path

    state_dict=torch.load([lastest_weight,load_pretrain_weights][config.specific_weights])
    
    model.load_state_dict(state_dict) 
    #model.load_state_dict(state_dict,strict=False) 

    #--------------------------------------------#
    #  將模型設至evaluation mode
    #--------------------------------------------#
    # Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    # 先前因為在'evaluation'重新載入模型所以沒有此問題
    # 但會導致模型實際上是在CPU進行evaluation
    # 若要在GPU上進行則需要將datasetloader輸出的image 在送入模型前使用todevice)
    #--------------------------------------------#
    
    model.eval()


    # 缺少batch_size時如何補上維度　　例如只有torch.Size([416, 36])
    # 使用 unsqueeze(0) 方法在输入数据的第一个维度添加一个批次维度。例如，如果输入数据形状是 [416, 36]，添加批次维度后，形状变为 [1, 416, 36]。
    # print(input_sequence)
    # print(input_sequence.shape)


    with torch.no_grad():
        pred = model(input_sequence.unsqueeze(0) ) # get predict probability

    print('raw_prediction logtis', pred, pred.shape, sep="\n")

    predicted_prob_max=pred.argmax(dim=1)
    print('Prediction: ', label_name_list[predicted_prob_max])
    return label_name_list[predicted_prob_max]

if __name__ == '__main__' :

    input_data_path  = r"dataset\WM_N02王臣睿A133595391＿20180710＿44w1d.mp4"
    origin_file_path = base_parameter (input_data_path) 
    vel_file_path    = calculate_velocity (origin_file_path) 
    acc_file_path    = calculate_acc (vel_file_path)  # preprocessing_dataset\output_WM_N02王臣睿A133595391＿20180710＿44w1d_2024_06_02_15_53_36\WM_N02王臣睿A133595391＿20180710＿44w1d_add_add_vel_acc.json

    shutil.copy(acc_file_path,config.output_path)
    write_xlsx(acc_file_path)

    print(f"finish processing {input_data_path}")

    Dir2JSON_file_path = Dir2JSON("preprocessing_dataset\output_WM_N02王臣睿A133595391＿20180710＿44w1d_2024_06_02_15_53_36\WM_N02王臣睿A133595391＿20180710＿44w1d_add_add_vel_acc.json").output_path_name()
    # print(Dir2JSON_file_path) # output\WM_N02王臣睿A133595391＿20180710＿44w1d_add_add_vel_acc_input.json
    
    input_sequence = dataset_getitem(Dir2JSON_file_path)
    output = model()
