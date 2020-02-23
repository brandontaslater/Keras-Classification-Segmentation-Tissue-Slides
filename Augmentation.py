import Augmentor
import os.path


# file path
ata_folder = os.path.join(r"C:\Users\YouWont4GetMe\Desktop\Dissertation\Images\Images for editing\Testing_Augmentation")

# number of images
number = 1

# augmentation to original
original = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
original.resize(probability=1.0, width=224, height=224)
original.sample(number)

# rotate90
r90 = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
r90.rotate90(probability=1.0)
r90.resize(probability=1.0, width=224, height=224)
r90.sample(number)

#rotate180
r180 = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
r180.rotate180(probability=1.0)
r180.resize(probability=1.0, width=224, height=224)
r180.sample(number)

#rotate270
r270 = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
r270.resize(probability=1.0, width=224, height=224)
r270.rotate270(probability=1.0)
r270.sample(number)

#flip_left_right
flip_left = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
flip_left.resize(probability=1.0, width=224, height=224)
flip_left.flip_left_right(probability=1.0)
flip_left.sample(number)

#resize
flip_up = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
flip_up.flip_top_bottom(probability=1.0)
flip_up.resize(probability=1.0, width=224, height=224)
flip_up.sample(number)

#skew_left_right
skew_left_right = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
skew_left_right.skew_left_right(probability=1.0)
skew_left_right.resize(probability=1.0, width=224, height=224)
skew_left_right.sample(number*2)

#random_distortion
p = Augmentor.Pipeline(source_directory=ata_folder, save_format="png")
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=6)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.resize(probability=1.0, width=224, height=224)
p.sample(25*number)
