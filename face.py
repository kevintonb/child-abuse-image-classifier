import tensorflow as tf
from subprocess import check_output
import sys
import os


def findFace():
	print("Finding Faces")
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	image_path_face = sys.argv[1]
	image_data_face = tf.gfile.FastGFile(image_path_face, 'rb').read()
	label_lines_face = [line.rstrip() for line 
	               in tf.gfile.GFile("face/tf_files/retrained_labels.txt")]
	with tf.gfile.FastGFile("face/tf_files/retrained_graph.pb", 'rb') as f:
	    graph_def_face = tf.GraphDef()   
	    graph_def_face.ParseFromString(f.read()) 
	    _ = tf.import_graph_def(graph_def_face, name='')
	with tf.Session() as sess:
	    softmax_tensor_face = sess.graph.get_tensor_by_name('final_result:0')
	    predictions_face = sess.run(softmax_tensor_face, \
	         {'DecodeJpeg/contents:0': image_data_face})
	    top_k_face = predictions_face[0].argsort()[-len(predictions_face[0]):][::-1]
	    #output
	    for node_id_face in top_k_face:
	        while (node_id_face==1):
	            human_string_face = label_lines_face[node_id_face]
	            score_face_1 = predictions_face[0][node_id_face]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_face_1 = ('%s (score = %.5f)' % (human_string_face,score_face_1))
	            print(out_face_1)
	            break
	        while (node_id_face==0):
	            human_string_face = label_lines_face[node_id_face]
	            score_face_0 = predictions_face[0][node_id_face]
	            #print('%s (score = %.5f)' % (human_string, score))
	            out_face_0 = ('%s (score = %.5f)' % (human_string_face,score_face_0))
	            print(out_face_0)
	            break
	print(max_value(score_face_1,score_face_0))
	return max_value(score_face_1,score_face_0)




def get_pid(name):
    return check_output(["pidof",name])



def max_value(a,b):
	if(a>b):
		print("Face Found")
		return a
		
	else:
		print("Face not found, exiting")
		exit()
	
findFace()