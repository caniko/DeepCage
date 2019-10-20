import concurrent.futures as futures

def try_multiple_operations(factor):
    return (5 + factor) * factor

executor = futures.ProcessPoolExecutor(10)
futures = [executor.submit(try_multiple_operations, x) 
           for x in range(10)]
futures.wait(futures)

for f in futures:
    print(f)
