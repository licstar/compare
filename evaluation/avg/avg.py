import multiprocessing
import os
import time

#models = ["hidden", "lbl", "senna"]
#models = ["rand"]
models = ["ivlblskip", "ivlblcbow"]

liblinear_dir = "liblinear-1.94"

def func(msg, vec_dir, ret_dir):
    vec_file = "%s/%s" % (vec_dir, msg)
    train_file = "%s_%s_train.txt" % (vec_dir, msg)
    test_file = "%s_%s_test.txt" % (vec_dir, msg)
    model_file = "%s.model" % train_file
    tmp_file = "%s_%s_out" % (vec_dir, msg)
    out_file = "%s/%s" % (ret_dir, msg)
    os.system("./avg_embedding %s imdb_train.txt imdb_test.txt %s %s" % (vec_file, train_file, test_file))
    os.system("%s/train %s" % (liblinear_dir, train_file))
    os.system("%s/predict %s %s %s > %s" % (liblinear_dir, train_file, model_file, tmp_file, out_file))
    os.system("%s/predict %s %s %s >> %s" % (liblinear_dir, test_file, model_file, tmp_file, out_file))
    os.system("rm %s" % train_file)
    os.system("rm %s" % test_file)
    os.system("rm %s" % model_file)
    os.system("rm %s" % tmp_file)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=20)

    for model in models:
        vec_dir = "vec_%s" % model
        ret_dir = "ret_avg2_%s" % model

        if not os.path.exists(ret_dir): 
            os.makedirs(ret_dir) 

        for lists in os.listdir(vec_dir):
            if not os.path.exists(os.path.join(ret_dir, lists)):

                x = lists.replace('.txt','').replace('.bz2','').split('_')
                #if not "v50" in lists:
                #    continue
                if int(x[-1]) > 10 and "10m" in lists and int(x[-1]) % 100 != 0:
                    continue
                if int(x[-1]) > 10 and "100m" in lists and int(x[-1]) % 10 != 0:
                    continue
                print lists
                pool.apply_async(func, (lists, vec_dir, ret_dir, ))
    pool.close()
    pool.join()
    print "Sub-process(es) done."
    
    
