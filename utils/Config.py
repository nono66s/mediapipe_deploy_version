

import os

class config:

    #-------------------------------------#
    #   Path
    #-------------------------------------#
    root_path                             = os.path.abspath("./")               # 專案 檔案夾
    dataset_path                          = "dataset\\"                         # 輸入 檔案夾
    preprocessing_path                    = "preprocessing_dataset\\"                         # 輸出 檔案夾
    output_path                           = "output\\"                         # 輸出 檔案夾

    Dataset_name                          = "baby_datasets"
    
    #-------------------------------------#
    #   Variable
    #-------------------------------------#
    body_landmark_name={
                        'NOSE'             : 0,
                        'LEFT_EYE_INNER '  : 1,
                        'LEFT_EYE'         : 2,
                        'LEFT_EYE_OUTER '  : 3,
                        'RIGHT_EYE_INNER ' : 4,
                        'RIGHT_EYE '       : 5,
                        'RIGHT_EYE_OUTER ' : 6,
                        'LEFT_EAR'         : 7,
                        'RIGHT_EAR '       : 8,
                        'MOUTH_LEFT '      : 9,
                        'MOUTH_RIGHT '     : 10,
                        'LEFT_SHOULDER '   : 11,
                        'RIGHT_SHOULDER '  : 12,
                        'LEFT_ELBOW '      : 13,
                        'RIGHT_ELBOW '     : 14,
                        'LEFT_WRIST '      : 15,
                        'RIGHT_WRIST '     : 16,
                        'LEFT_PINKY '      : 17,
                        'RIGHT_PINKY '     : 18,
                        'LEFT_INDEX '      : 19,
                        'RIGHT_INDEX '     : 20,
                        'LEFT_THUMB '      : 21,
                        'RIGHT_THUMB '     : 22,
                        'LEFT_HIP '        : 23,
                        'RIGHT_HIP'        : 24,
                        'LEFT_KNEE '       : 25,
                        'RIGHT_KNEE '      : 26,
                        'LEFT_ANKLE'       : 27,
                        'RIGHT_ANKLE '     : 28,
                        'LEFT_HEEL'        : 29,
                        'RIGHT_HEEL '      : 30,
                        'LEFT_FOOT_INDEX ' : 31,
                        'RIGHT_FOOT_INDEX ': 32,
                        }
    #----------------------------------------------------------------#
    # 圖像縮放、輸入大小
    #----------------------------------------------------------------#   

    seq_length, input_size = 2997, 113
    
    #----------------------------------------------------------------#
    # 預訓練參數位置
    # 包含從網絡下載的以及接續訓練
    #----------------------------------------------------------------#   

    use_pretrain_weight  = [True,False][0]


    #----------------------------------------------------------------#
    # evaluation 時
    # specific_weights
    # 0 -> 使用最後最新訓練的權重
    # 1 -> 使用自定義的權重(例如預訓練)
    #----------------------------------------------------------------#

    specific_weights=0
    specific_weights_path=os.path.join('./model_data/','resnet101_caffe.pth')


    def __init__(self):
        pass

    def display(self):                                                                      # 顯示上面所有訊息!
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):                   #確保打應出來的是不帶底線、不可呼叫
                print("{:30} {}".format(a, getattr(self, a)))                               #getattr(object, name)獲取對象的屬性並轉為字符串
        print("\n")

# test = config() # To use the class, first create an instance
# test.display()