import tensorflow as tf
import sys
import os
import psutil
def findKid():
	print("Finding Kids..")
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	image_path_kid = sys.argv[1]
	image_data_kid = tf.gfile.FastGFile(image_path_kid, 'rb').read()
	label_lines_kid = [line.rstrip() for line 
	               in tf.gfile.GFile("kid/tf_files/retrained_labels.txt")]
	with tf.gfile.FastGFile("kid/tf_files/retrained_graph.pb", 'rb') as f:
	    graph_def_kid = tf.GraphDef()   
	    graph_def_kid.ParseFromString(f.read()) 
	    _ = tf.import_graph_def(graph_def_kid, name='')
	with tf.Session() as sess:
	    softmax_tensor_kid = sess.graph.get_tensor_by_name('final_result:0')
	    predictions_kid = sess.run(softmax_tensor_kid, \
	         {'DecodeJpeg/contents:0': image_data_kid})
	    top_k_kid = predictions_kid[0].argsort()[-len(predictions_kid[0]):][::-1]
	    #output
	    for node_id_kid in top_k_kid:
	        while (node_id_kid==1):
	            human_string_kid = label_lines_kid[node_id_kid]
	            score_kid_1 = predictions_kid[0][node_id_kid]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_kid_1 = ('%s (score = %.5f)' % (human_string_kid,score_kid_1))
	            print(out_kid_1)
	            break
	        while (node_id_kid==0):
	            human_string_kid = label_lines_kid[node_id_kid]
	            score_kid_0 = predictions_kid[0][node_id_kid]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_kid_0 = ('%s (score = %.5f)' % (human_string_kid,score_kid_0))
	            print(out_kid_0)
	            break
	print(max_value(score_kid_1,score_kid_0))
	#os.remove("kid.pyc")


def max_value(a,b):
	if(a>b):
		print("Kid not found ! ")
		return a
		
		
	else:
		print("Kid found")
		sys.exit()


findKid()	