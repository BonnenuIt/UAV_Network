from matplotlib import pyplot as plt


def MyAnimation(num_frame=200, root_path='imgs/ma', save_path='cosxt.gif'):
    # https://zhuanlan.zhihu.com/p/106283237
    import matplotlib.animation as animation
    import cv2

    fig = plt.figure()
    ims = []
    for i in range(num_frame):
        # 用opencv读取图片
        img = cv2.imread(root_path+str(i+1)+'.png')
        (r, g, b) = cv2.split(img)  
        img = cv2.merge([b,g,r])
        im = plt.imshow(img, animated=True)
        plt.axis('off')
        # plt.show()
        ims.append([im])
    # 用animation中的ArtistAnimation实现动画. 每帧之间间隔500毫秒, 每隔1000毫秒重复一次,循环播放动画.
    ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True, repeat_delay=100)

    ### 保存动态图片
    ani.save(save_path, fps=20)  

if __name__ == '__main__':
    MyAnimation(num_frame=388, root_path='results/imgs/ma', save_path='results/gifs/41.gif')