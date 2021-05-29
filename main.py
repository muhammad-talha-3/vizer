# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import cv2 
import datetime
import random
from flask import Flask, render_template,request,flash,send_file,redirect,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

home_images = []
social_imgs=[]
site_imgs = {}
LOGO = 'images/1.png'
cover_img = 'images/2.png'
home_images.append('images/power-full-icon.png')
home_images.append('images/optimised-icon.png')
home_images.append('images/headfone-icon.png')
home_images.append('images/icon-4.png')
home_images.append('images/icon-5.png')
home_images.append('images/icon-6.png')

social_imgs.append('images/fb-icon.png')
social_imgs.append('images/twitter-icon.png')
social_imgs.append('images/in-icon.png')
social_imgs.append('images/google-icon.png')

site_imgs['logo'] = LOGO
site_imgs['cover_img'] = cover_img
site_imgs['built_icon'] = 'images/bulit-icon.png'
site_imgs['social_imgs'] = social_imgs
site_imgs['right_sec'] = 'images/img-2.png'

# print(type(site_imgs))
# print("hash value: %032x" % hash)      
	

def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    frames = []
    
    while vidObj.isOpened():
#     while count<duration: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        if success ==False:
            break
        # Saves the frames with frame-count 
#         cv2.imwrite("images//frame %d.jpg" %count, image) 
        frames.append(image)
       # cv2.imshow("frame%d.jpg" %count, image) 
        count += 1
    vidObj.release()
    print("number of frames: ",count)
    return count,frames

def del_random_frames(images):
	prob = 0.5
	to_del = []
	sampleList = range(len(images))
	for i in range(int(len(images)*prob)):
	    x = random.choice(sampleList)
	    to_del.append(x)
	print(to_del)

	for index in sorted(to_del, reverse=True):
	    del images[index]
	return images

def del_frames_by_model(images,selected_cat):
	""" this method gets input all frames of video and category selected and delete related frames selected from category """
	model = load_model('model.h5')
	to_del = []
	label = 0
	for i in range(len(images)):
		image = cv2.resize(images[i],(224,224))
		img_batch = np.expand_dims(image,axis = 0)
		img_preprocessed = preprocess_input(img_batch)

		result = model.predict(img_preprocessed)
		result = result.tolist()
		maxValue = max(result[0])
		label = str(result[0].index(maxValue))

		# print("Label : " + str(label) + " Selected Cat : " + str(selected_cat) )
		
		if label == selected_cat:
			to_del.append(i)
		# print("Del List : ", to_del)
		print("Frame " + str(i) + " processed")

	# delete frames
	for index in sorted(to_del, reverse=True):
	    del images[index]
	# return remaining frames
	return images

def creat_video(images,hash):
	video_name = 'saved_video/'+str(hash)+'.mp4'
	frame = images[0]
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'),30,(width,height))

	for image in images:
	#     video.write(cv2.imread(os.path.join(image_folder, image)))
	    video.write( image)

	# cv2.destroyAllWindows()
	video.release()

main = Flask(__name__, template_folder='templates')
main.secret_key = "abc" 


@main.route('/')
@main.route('/home')
def home():
    return render_template('index.html',images = site_imgs,home_imgs = home_images)


@main.route('/video')
def video():
    return render_template('video.html',error = "",images=site_imgs)


@main.route('/',methods = ["POST"])
def upload():

	hash = random.getrandbits(128)
	if 'video' not in request.files:
		error = "No file part"
		return render_template('video.html',error=error,images = site_imgs)
	        # flash('No file part')
	        # return redirect(request(url_for(home)))

	file = request.files['video']
	if file.filename == '':
		error = "No video File Selected"
		return render_template('video.html',error=error,images = site_imgs)
	        # flash('No selected file')
	        # return redirect(request(url_for(home)))
	
	# category selected from menu

	category = request.form.get('category')
	char_image = request.files['image']
	# print(type(file))
	file.save(os.path.join('saved_video/%032x.mp4' % hash))

	# optional
	# clip = cv2.VideoCapture('saved_video/%032x.mp4' % hash)
	# frames = clip.get(cv2.CAP_PROP_FRAME_COUNT)
	# fps = int(clip.get(cv2.CAP_PROP_FPS))
	  
	# # calculate dusration of the video
	# seconds = int(frames / fps)
	# video_time = str(datetime.timedelta(seconds=seconds))
	# print("duration in seconds:", seconds)
	# print("video time:", video_time)
	# print("category selected is: ",category)
	# optional end

	count,images = FrameCapture('saved_video/%032x.mp4' % hash)

	# randomly deleting frames this part will be replced by model
	# images = del_random_frames(images)

	# deleting frames using model
	images = del_frames_by_model(images,category)

	# print("Total Frames: ",count)
	# print("Frames Deleted: ",count - len(images))
	# print("Remaining frames: ",len(images))
	# print("Congratulations! Your video is ready. Download it")

	# creating video out of frames
	creat_video(images,hash)
	# os.listdir(image_folder)
	return render_template('download.html',images = site_imgs,name = hash)
	# return file.filename

@main.route('/download/<string:hash>',methods = ['GET'])
def download(hash):
	return send_file('saved_video/'+str(hash)+'.mp4',as_attachment = True)

if __name__ == '__main__':
    # app = Flask(__name__, template_folder='templates')
    print(type(site_imgs))
    print(site_imgs['logo'])
    main.run()

