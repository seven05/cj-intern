import time
import time_module


input = "tving_video_224/P470472958_EPI000"
for i in range(1, 10):
    start = time.time()

    a, b, c, d = time_module.main(input)
    end = time.time()
    print(a, b, c, d, end-start)