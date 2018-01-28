import tensorflow as tf
import sys
import os
import cv2

def findAbuse():
	print("Finding Abusive Image")
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	image_path_abuse = sys.argv[1]
	image_data_abuse = tf.gfile.FastGFile(image_path_abuse, 'rb').read()
	label_lines_abuse = [line.rstrip() for line 
	               in tf.gfile.GFile("abuse/tf_files/retrained_labels.txt")]
	with tf.gfile.FastGFile("abuse/tf_files/retrained_graph.pb", 'rb') as f:
	    graph_def_abuse = tf.GraphDef()   
	    graph_def_abuse.ParseFromString(f.read()) 
	    _ = tf.import_graph_def(graph_def_abuse, name='')
	with tf.Session() as sess:
	    softmax_tensor_abuse = sess.graph.get_tensor_by_name('final_result:0')
	    predictions_abuse = sess.run(softmax_tensor_abuse, \
	         {'DecodeJpeg/contents:0': image_data_abuse})
	    top_k_abuse = predictions_abuse[0].argsort()[-len(predictions_abuse[0]):][::-1]
	    #output
	    for node_id_abuse in top_k_abuse:
	        while (node_id_abuse==1):
	            human_string_abuse = label_lines_abuse[node_id_abuse]
	            score_abuse_1 = predictions_abuse[0][node_id_abuse]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_abuse_1 = ('%s (score = %.5f)' % (human_string_abuse,score_abuse_1))
	            print(out_abuse_1)
	            break
	        while (node_id_abuse==0):
	            human_string_abuse = label_lines_abuse[node_id_abuse]
	            score_abuse_0 = predictions_abuse[0][node_id_abuse]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_abuse_0 = ('%s (score = %.5f)' % (human_string_abuse,score_abuse_0))
	            print(out_abuse_0)
	            break
	print(max_value(score_abuse_1,score_abuse_0))


def blur_img(img):
	blurImg = cv2.blur(img,(50,50))



def max_value(a,b):
	if(a>b):
		print("Abusive content found")
		return a
		
	else:
		print("Abusive content not found :)")
		return b
findAbuse()