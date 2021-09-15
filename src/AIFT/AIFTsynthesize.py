import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        zca_whitening=True,
        fill_mode='nearest')


# Change directory name to the one that has all the images you would like to synthesize (search folder)
count = 1
for file in os.listdir("search"):
	if file.endswith(".jpg"):
		img = load_img('search/' + file)  # this is a PIL image
		if count % 200 == 1:
			print(count)
		count += 1
		# resize images to 1.0, 1.2, and 1.5
		img2 = img.resize((77, 77), resample=0)
		img3 = img.resize((96, 96), resample=0)
		
		img = img_to_array(img)
		img = img.reshape((1,) + img.shape)
		img_flow = datagen.flow(img, batch_size=1)
		for i, new_imgs in enumerate(img_flow):
			new_img = array_to_img(new_imgs[0], scale=True)
			f_name = file[:-4]
			f_name = 'search/' + f_name + "_" + str(i + 1) + ".jpg"
			new_img.save(f_name)
			if i >= 4:
				break
				
		img = img_to_array(img2)
		img = img.reshape((1,) + img.shape)
		img_flow = datagen.flow(img, batch_size=1)
		for i, new_imgs in enumerate(img_flow):
			if i > 4:
				new_img = array_to_img(new_imgs[0], scale=True)
				
				# resize it back to 64x64 here
				new_img = new_img.resize((64, 64), resample = 0)
				
				f_name = file[:-4]
				f_name = 'search/' + f_name + "_" + str(i + 1) + ".jpg"
				new_img.save(f_name)
			if i >= 9:
				break
				
		img = img_to_array(img3)
		img = img.reshape((1,) + img.shape)
		img_flow = datagen.flow(img, batch_size=1)
		for i, new_imgs in enumerate(img_flow):
			if i > 9:
				new_img = array_to_img(new_imgs[0], scale=True)
				
				# resize it back to 64x64 here
				new_img = new_img.resize((64, 64), resample = 0)
				
				f_name = file[:-4]
				f_name = 'search/' + f_name + "_" + str(i + 1) + ".jpg"
				new_img.save(f_name)
			if i >= 14:
				break
