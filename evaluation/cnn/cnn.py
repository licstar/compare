import multiprocessing
import os
import time

#models = ["hidden", "lbl", "senna"]
models = ["ivlblskip", "ivlblcbow"]
#models = ["rand"]

def func(msg, vec_dir, ret_dir):
    vec_file = "%s/%s" % (vec_dir, msg)
    out_file = "%s/%s" % (ret_dir, msg)
    for i in range(0, 5):
        os.system("./cnn_senna %s tree_train.txt tree_test.txt 5 %d tree_dev.txt 5 90 >> %s" % (vec_file, i, out_file))
    #os.system("./cnn_senna %s tree_train.txt tree_test.txt 5 1 tree_dev.txt 5 90 >> %s" % (vec_file, out_file))
    #os.system("./cnn_senna %s tree_train.txt tree_test.txt 5 2 tree_dev.txt 5 90 >> %s" % (vec_file, out_file))
    #os.system("./cnn_senna %s tree_train.txt tree_test.txt 5 3 tree_dev.txt 5 90 >> %s" % (vec_file, out_file))
    #os.system("./cnn_senna %s tree_train.txt tree_test.txt 5 4 tree_dev.txt 5 90 >> %s" % (vec_file, out_file))




if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=1)

    for model in models:
        vec_dir = "vec_%s" % model
        ret_dir = "ret_cnn_%s" % model

        if not os.path.exists(ret_dir):
            os.makedirs(ret_dir)
        #func('50_2_ns5_16')
        
        for lists in os.listdir(vec_dir):
            if not os.path.exists(os.path.join(ret_dir, lists)):

                x = lists.replace('.txt','').replace('.bz2','').split('_')
                #if not "v50" in lists:
                #    continue
                iter = int(x[-1])
                if not (iter == 1 or iter == 3 or iter == 5 or iter == 20 or iter == 10 or iter == 33 or iter == 100 or iter == 1000 or iter == 10000):
                    continue
                print lists
                pool.apply_async(func, (lists, vec_dir, ret_dir, ))
    pool.close()
    pool.join()
    print "Sub-process(es) done."
    
    
