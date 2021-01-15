
from os import getcwd

from LBPfuncs import CompareLBP, MyResize
from UpdateFuncs import *
from get_LBP_from_Image import *

def update(front, back):
    #sysbg = cv2.createBackgroundSubtractorMOG2(500, 30, detectShadows=True) # 构造高斯混合模型，阴影检测
    """文件的读取与视频文件的初始化"""
    c_path = getcwd()
    vid = cv2.VideoCapture(front)
    fps = vid.get(cv2.CAP_PROP_FPS)
    CheckVideo(vid)
    
    vid2 = cv2.VideoCapture(back)
    CheckVideo(vid2)
    
    h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    """各类参数与过程中用到的计算用矩阵"""
    a = 0.055  # 更新率，a为背景更新率，b为前景更新率
    b = 0.001
    BinaryThreshold = 30
    LBP_threshold = 0.5
    decay = 0.0000001
    UpdateThred = 20
    DelayWaitFrameNum = 80
    NumFrameForceForeToBack = 100  # 当一个目标多少帧之后，它会强制转换为背景
    kernel1 = np.ones((1, 1), np.uint8) # 形态学处理
    kernel2 = np.ones((2, 2), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel4 = np.ones((3, 3), np.uint8)
    chn = 3  # 色彩通道数
    w = 256
    h = 256
    shapewh = (np.int(w), np.int(h), chn)
    shapehw = (np.int(h), np.int(w), chn)
    sizewh = (np.int(w), np.int(h))
    sizehw = (np.int(h), np.int(w))
    vw= cv2.VideoWriter('img/output.avi', fourcc, fps, sizewh)
    """过程中承载参数的矩阵的设置"""
    BG = np.zeros(shape=shapehw, dtype=np.uint8)
    Sub = np.ones(shape=shapehw, dtype=np.uint8)
    SubAll = np.zeros(shape=sizehw, dtype=np.uint8)
    ColorSubShow = np.ones(shape=shapehw, dtype=np.uint8)
    SubR = np.ones(shape=shapehw, dtype=np.uint8)
    SubTemp = np.zeros(shape=shapehw, dtype=np.uint8)
    ForeFlag = np.zeros(shape=shapehw, dtype=np.int32)
    FlagOld = np.zeros(shape=shapehw, dtype=np.int32)
    LongNotGrowing = np.zeros(shape=shapehw, dtype=np.int32)
    DelayFlag = np.zeros(shape=shapehw, dtype=np.int32)
    SubInInt32 = np.zeros(shape=shapehw, dtype=np.int32)
    BG_DiffFlag = np.zeros(shape=shapehw, dtype=np.uint8)
    """显示窗口初始化"""
    #cv2.namedWindow("Current Frame (Colored)", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("Current Background (Colored)", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("Sub (Colored)", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("Sub with All color channel merged", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("Region Flaged as Backround in Current Frame", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("Region Flaged as Foreground in Current Frame", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("contours", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("System", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("ForeFlag", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("LBP", cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow("LBP Fore", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Video synthesis", cv2.WINDOW_KEEPRATIO)
    """背景初始化，采用第一帧初始化"""
    ret, initialframe = vid.read()
    initialframe = cv2.resize(initialframe, sizewh, interpolation=cv2.INTER_CUBIC)
    FrameNum = 0  # 当前帧计数器
    BG = initialframe
    """各种更新时的策略选择"""
    updateAsAll = True  # 按照红绿蓝三色更新
    UpdateSeparately = True  # 是否将前后背景分开更新
    EliminateForegroundTooLong = True  # 定期消除长时间占用前景像素
    UseMinimumRecContours = False  # 使用最小矩形框选选择目标。
    UpdateWithinContours = True  # 以轮廓内物体为前景更新
    DoMorphology_1 = True  # 使用形态学处理消除小区域，先开后闭
    DoMorphology_2 = True  # 获得轮廓后使用形态学处理消除小区域，先开后闭。
    GenContours = True  # 显示物体轮廓
    MedBlur = True  # 中值滤波选择。
    UseLBP = True  # 使用LBP纹理信息
    CheckTackle(GenContours, UseMinimumRecContours, UpdateWithinContours, UpdateSeparately, a, b)
    lbp = LBP()
    """视频目标追踪循环开始"""
    while 1:
        FrameNum += 1
        ret, frame_org = vid.read()
        ret2, frame_org2 = vid2.read()
        if not ret or not ret2:
            print("没有正确读取视频")
            break
        frame_org = cv2.resize(frame_org, sizewh, interpolation=cv2.INTER_CUBIC)
        frame = frame_org[:, :, 0:chn]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayBG = cv2.cvtColor(BG, cv2.COLOR_BGR2GRAY) # 背景
    
        frame_org2 = cv2.resize(frame_org2, sizewh, interpolation=cv2.INTER_CUBIC)
        frame2 = frame_org2[:, :, 0:chn]
        # ShowSubIm("Current Frame and BG", frame_org, grayBG)
    
        # lbphist = lbp.lbp_basic(gray)
        # cv2.imshow("LBP", lbphist)
        # lbp.show_basic_hist(lbphist)
        # plt.show()
        #SystemSub = sysbg.apply(frame) # 使用高斯混合模型
        #cv2.imshow("System", SystemSub)
        Sub1 = cv2.absdiff(BG, frame) # 获取与背景的差异
        ret, Sub = cv2.threshold(Sub1, BinaryThreshold, 255, type=cv2.THRESH_BINARY) # 高于阈值像素设为255
        """是否清除过长的前景"""
        if FrameNum > 0:
            if EliminateForegroundTooLong:
                SubInInt32 = np.where(Sub.copy() < 1, 0, 1) # 阈值化后的图像设为0，1图像
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
        Old_BG = BG.copy()
        """是否做形态学处理"""
        if DoMorphology_1:
            Sub = cv2.morphologyEx(Sub, cv2.MORPH_OPEN, kernel1)  # 开
            Sub = cv2.morphologyEx(Sub, cv2.MORPH_CLOSE, kernel2)  # 闭
        ColorSubShow = Sub.copy()
        SubAll = Sub[:, :, 0] + Sub[:, :, 1] + Sub[:, :, 2]  # 统一三个色彩通道的前景探测结果
        SubAll = np.where(SubAll < 1, SubAll, 255)
        """选择是否做中值滤波消除噪声干扰"""
        if MedBlur:
            Sub = cv2.medianBlur(Sub, 3)
        """生成轮廓"""
        if GenContours:
            contours, hierarchy = cv2.findContours(image=SubAll, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # 检测轮廓
            FrameForContours = frame.copy()
            if UseMinimumRecContours:  # 生成最小矩阵轮廓
                SubTemp = np.where(SubTemp == 0, SubTemp, 0)
                for i, contour in enumerate(contours):
                    rct = cv2.minAreaRect(contour) # 最小矩形区域
                    box = cv2.boxPoints(rct)
                    box = np.int0(box)
                    cv2.drawContours(FrameForContours, [box], 0, (255, 255, 255), -1)
                    cv2.drawContours(SubTemp, [box], 0, (255, 255, 255), -1)
            else:  # 使用不规则轮廓
                SubTemp = np.where(SubTemp == 0, SubTemp, 0)
                for i, contour in enumerate(contours):
                    cv2.drawContours(FrameForContours, contours, i, (255, 255, 255), -1)
                    cv2.drawContours(SubTemp, contours, i, (255, 255, 255), -1)
            #cv2.imshow("contours", FrameForContours)
            """判断是否使用轮廓内的内容进行升级"""
            if UpdateWithinContours:
                Sub = np.where(SubTemp < 1, Sub, 255)
                if DoMorphology_2:
                    Sub = cv2.morphologyEx(Sub, cv2.MORPH_OPEN, kernel3)  # 开
                    Sub = cv2.morphologyEx(Sub, cv2.MORPH_CLOSE, kernel4)  # 闭
        """如果不使用轮廓升级"""
        if not UpdateWithinContours:  # 不使用轮廓
            """判断是否使用全部色彩层信息进行背景升级"""
            if updateAsAll:
                for i in range(chn):
                    Sub[:, :, i] = SubAll
        """LBP处理背景掩膜控制更新"""
        SubR = np.uint8(Sub / 255)  # SubR用来做掩模版，区分出前景和后景
        # out = CompareLBP(cv2.pyrDown(gray), cv2.pyrDown(grayBG), cv2.pyrDown(SubAll / 255), windowSize=8, step=8)
        if not FrameNum % 1:
            if UseLBP:
                out = CompareLBP(MyResize(gray, 1.5), MyResize(grayBG, 1.5), MyResize(SubAll / 255, 1.5), windowSize=7,
                                 step=7, region_thresh=2, decay=decay) # 比较当前帧与背景帧纹理信息
    
                out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.1,
                                       sigmaY=1.1) # 高斯滤波
                #cv2.imshow("LBP", out)
                out = cv2.resize(out, (SubR.shape[0], SubR.shape[1]))
                ret, out = cv2.threshold(out, LBP_threshold, 1, type=cv2.THRESH_BINARY)
                for i in range(chn):
                    SubR[:, :, i] = SubR[:, :, i] + out
                SubR = np.uint8(np.where(SubR > 0, 1, 0))
                #cv2.imshow("LBP Fore", np.uint8(out * gray))
        BG_ForeD = BG * SubR
        BG_BackD = BG - BG_ForeD
        #cv2.imshow("Region Flaged as Background in Current Background", BG_BackD[:, :, 0])
        #cv2.imshow("Region Flaged as Foreground in Current Background", BG_ForeD[:, :, 0])
        frame_fore = frame * SubR
        frame_back = frame - frame_fore
        #cv2.imshow("Region Flaged as Backround in Current Frame", frame_back[:, :, 0])
        #cv2.imshow("Region Flaged as Foreground in Current Frame", frame_fore[:, :, :])
        BG_Back = cv2.addWeighted(BG_BackD, 1 - a, frame_back, a, 0) # 图像叠加更新背景
        BG_Fore = cv2.addWeighted(BG_ForeD, 1 - b, frame_fore, b, 0)
        BG = BG_Back + BG_Fore
        """控制背景更新率"""
        BG_Diff = cv2.absdiff(Old_BG, BG)
        BG_DiffFlag = np.where(BG_Diff < UpdateThred, BG_DiffFlag, 1)
        Old_BG_Reserve = Old_BG * BG_DiffFlag
        BG_NotUpdate = BG * BG_DiffFlag
        BG_Update = BG - BG_NotUpdate
        BG = BG_Update + Old_BG_Reserve
        """显示各个图像"""
        #cv2.imshow('Current Frame (Colored)', frame)
        #cv2.imshow("Current Background (Colored)", BG)
        #cv2.imshow("Sub (Colored)", ColorSubShow)
        #cv2.imshow("Sub with All color channel merged", SubAll)

        Sub_copy = Sub.copy()
        Sub_copy[Sub_copy>0 ] = 1 # mask
        frame_front = Sub_copy * frame # 前景
        #cv2.imshow("Foreground", frame_front) 
        #print(Sub)

        """视频合成"""
        bg_mask = Sub_copy.copy()
        bg_mask[bg_mask < 1] = 2
        bg_mask[bg_mask == 1] = 0
        bg_mask[bg_mask > 0] = 1
        frame_bg = bg_mask * frame2
        frame_add = cv2.add(frame_front, frame_bg)
        cv2.imshow("Video synthesis", frame_add)
        vw.write(frame_add)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    vid2.release()
    cv2.destroyAllWindows()
