import multiprocessing
import os
import time

#models = ["hidden", "lbl", "senna"]
#models = ["turian"]
models = ["ivlblskip", "ivlblcbow"]

def func(msg, vec_dir, ret_dir):
    arg = './ws %s/%s > %s/%s' % (vec_dir, msg, ret_dir, msg)
    print arg
    os.system(arg)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=16)

    for model in models:
        vec_dir = "vec_%s" % model
        ret_dir = "ret_ws_%s" % model

        if not os.path.exists(ret_dir):
            os.makedirs(ret_dir)

        for lists in os.listdir(vec_dir):
            if not os.path.exists(os.path.join(ret_dir, lists)):

                x = lists.replace('.txt','').replace('.bz2','').split('_')
                #if not "v50" in lists:
                #    continue
                #if int(x[-1]) > 10 and ("10m" in lists or "13m" in lists) and int(x[-1]) % 100 != 0:
                #    continue
                #if int(x[-1]) > 10 and "100m" in lists and int(x[-1]) % 10 != 0:
                #    continue
                print lists
                pool.apply_async(func, (lists, vec_dir, ret_dir, ))
    pool.close()
    pool.join()
    print "Sub-process(es) done."
    
    
