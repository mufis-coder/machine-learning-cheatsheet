# Thread, Batch, and TQDM

## Thread and batch

```py
from tqdm import tqdm
import time
import concurrent.futures
from more_itertools import grouper

start = time.time()

def convert_heic_png(data_list, global_count):
    data_list = list(filter(lambda item: item is not None, data_list))
    
    count = 0
    for data in tqdm(data_list, total=len(data_list)):
        do_something(data)
        count += 1

if __name__ == '__main__':
    datas = []

    # batching
    num_batch = int(len(datas)/1.5)
    batchs = [group for group in grouper(num_batch, datas)]

    # multiprocess
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    futures = [executor.submit(convert_heic_png, batchs[i], i)
               for i in range(0, len(batchs))]
    concurrent.futures.wait(futures)

    end = time.time()
    print(f"Total waktu: {(end - start)}")
```
