import cv2
import numpy as np


def CheckVideo(vid):
    flag = vid.isOpened()
    if flag:
        print("打开视频成功")
    else:
        print("打开视频失败")


def EliminateForeGround(Sub, ForeFlag, FlagOld, NumFrameForceForeToBack, LongNotGrowing, DelayFlag, DelayWaitFrameNum):
    SubInInt32 = np.where(Sub.copy() < 1, 0, 1)
    ForeFlag = ForeFlag + SubInInt32
    NotGrowing = np.where(FlagOld == ForeFlag, 0, 1)
    InvNotGrowing = np.where(FlagOld == ForeFlag, 1, 0)
    NotGrowing = np.where(ForeFlag >= NumFrameForceForeToBack, 1, NotGrowing)
    LongNotGrowing = LongNotGrowing + InvNotGrowing
    ForeFlag = ForeFlag * np.where(LongNotGrowing > 20, 0, 1)
    LongNotGrowing = np.where(LongNotGrowing > 20, 0, LongNotGrowing)
    Sub = np.where(LongNotGrowing <= 20, Sub, 0)
    Sub = np.where(ForeFlag < NumFrameForceForeToBack, Sub, 0)  # 清除存在超过NumFrameForceForeToBack帧的前景
    DelayFlag = np.where(ForeFlag >= NumFrameForceForeToBack, DelayFlag + 1, DelayFlag)
    ForeFlag = np.where(DelayFlag > DelayWaitFrameNum, 0, ForeFlag)
    DelayFlag = np.where(DelayFlag > DelayWaitFrameNum, 0, DelayFlag)
    #cv2.imshow("ForeFlag", np.uint8(ForeFlag))
    FlagOld = ForeFlag


def CheckTackle(GenContours, UseMinimumRecContours, UpdateWithinContours, UpdateSeparately, a, b):
    if not GenContours: # 设置为不显示轮廓
        if UseMinimumRecContours: # 最小矩形框
            UseMinimumRecContours = False
            print("未生成轮廓，以最小矩形轮廓更新被取消")
        if UpdateWithinContours: # 轮廓内物体为前景
            UpdateWithinContours = False
            print("未生成轮廓，以轮廓更新被取消")
    else:
        if UseMinimumRecContours:
            if not UpdateWithinContours:
                print("最小矩形轮廓被标出，更新背景时未使用")
        else:
            if not UpdateWithinContours:
                print("不规则轮廓被标出，更新背景时未使用")
    if not UpdateSeparately: # 前后景不分开处理
        b = a